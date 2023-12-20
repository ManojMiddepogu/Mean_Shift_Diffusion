import copy
import functools
import numpy as np
import os
import wandb

import torch.nn.functional as F
import blobfile as bf
from PIL import Image
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
from torchvision.utils import make_grid

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler, LossSecondMomentResampler, SameTSampler, AlternateSampler, GANTypeSampler
from .script_util import NUM_CLASSES
from .clustered_model import ClusteredModel
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
from .fid_score import calculate_activation_statistics, calculate_frechet_distance
from .inception import InceptionV3

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        fid_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        no_guidance_step=200000000,
        freeze_guidance_after_no_guidance_step=True,
        # Sampling arguments for visualization during training
        clip_denoised=True,
        num_samples=400,
        num_samples_batch_size=200,
        num_samples_visualize=100,
        use_ddim=False,
        image_size=64,
        training_data_inception_mu_sigma_path="",
        use_wandb=True,
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.fid_interval = fid_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.no_guidance_step = no_guidance_step
        self.freeze_guidance_after_no_guidance_step = freeze_guidance_after_no_guidance_step

        self.clip_denoised = clip_denoised
        self.num_samples = num_samples
        self.num_samples_batch_size = num_samples_batch_size
        self.num_samples_visualize = num_samples_visualize
        self.use_ddim = use_ddim
        self.image_size = image_size
        self.use_wandb = use_wandb

        self.training_data_inception_mu_sigma_path = training_data_inception_mu_sigma_path
        self.training_data_inception_mu = None
        self.training_data_inception_sigma = None
        self.inception_model = None
        if self.training_data_inception_mu_sigma_path != "":
            with np.load(self.training_data_inception_mu_sigma_path) as f:
                self.training_data_inception_mu, self.training_data_inception_sigma = f['mu'][:], f['sigma'][:]
                print("Loaded mu and sigma parameteres for FID calculation!")
            
            if dist.get_rank() == 0:
                block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
                self.inception_model = InceptionV3([block_idx]).to(dist_util.dev())
                print("Loaded Inception model for FID calculation!")

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)
    
    def _create_image_collage(self, samples, rows=8, cols=8):
        # Assuming samples is a list of numpy arrays with shape (H, W, C)
        h, w, c = samples[0].shape
        collage_width = cols * w
        collage_height = rows * h
        collage = Image.new('RGB', (collage_width, collage_height))

        # Paste images into collage
        for i, np_img in enumerate(samples):
            img = Image.fromarray(np_img.astype('uint8'))
            # Calculate position of current image
            row = i // cols
            col = i % cols
            position = (col * w, row * h)
            collage.paste(img, position)
            if i >= rows*cols - 1:  # Break after filling in rows x cols images
                break

        return collage
    
    def _plot_multiple_gaussian_contours(self, means, sigmas, plot_t):
        # Create a row of subplots
        fig, axes = plt.subplots(nrows=1, ncols=11, figsize=(11 * 4, 4))

        for i in range(0, means.shape[0], NUM_CLASSES):
            mu_batch = means[i:i + NUM_CLASSES]
            sigma_batch = sigmas[i:i + NUM_CLASSES]

            # Convert inputs to numpy arrays if they are not already
            mu_batch = np.array(mu_batch.view(NUM_CLASSES, -1).to('cpu'))
            sigma_batch = np.array(sigma_batch.view(NUM_CLASSES, -1).to('cpu'))
            reduced_means = PCA(n_components=2).fit_transform(mu_batch)

            ax = axes[i // NUM_CLASSES]
            # Plot circles for each Gaussian distribution
            for index, (mean, sigma) in enumerate(zip(reduced_means, sigma_batch)):
                ellipse = Ellipse(xy=mean, width=6 * sigma, height=6 * sigma, edgecolor='r', fc='None', lw=2)
                ax.add_patch(ellipse)
                ax.text(mean[0], mean[1], str(index), color='black', ha='center', va='center', fontsize=10)

            # Setting the limits of the plot
            ax.set_xlim(np.min(reduced_means) - 6*np.max(sigma_batch), np.max(reduced_means) + 6*np.max(sigma_batch))
            ax.set_ylim(np.min(reduced_means) - 6*np.max(sigma_batch), np.max(reduced_means) + 6*np.max(sigma_batch))

            # Add grid, labels, and title to each subplot
            ax.grid(True)
            ax.set_xlabel('X-axis')
            ax.set_ylabel('Y-axis')
            ax.set_title(f'Gaussian {plot_t[i].item()}')

        plt.tight_layout()
        return fig

    def run_loop(self):
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            batch, cond = next(self.data)
            self.run_step(batch, cond)
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0:
                self.save()

                # Sample num_samples_visualize images everytime we save the model
                if dist.get_rank() == 0:  # Make sure only the master process does the sampling
                    self.ddp_model.eval()

                    wandb_log_images = {}

                    with th.no_grad():
                        # Generate samples
                        fid_samples = None
                        run_num_samples = self.num_samples if (self.step % self.fid_interval == 0) else self.num_samples_visualize
                        run_samples_batch_size = self.num_samples_batch_size if (self.step % self.fid_interval == 0) else self.num_samples_visualize
                        for i in range(0, run_num_samples // run_samples_batch_size):
                            print(f"Sampling {run_num_samples} Images for batch number {i} out of {run_num_samples // self.num_samples_batch_size} batches!")
                            sample_fn = (
                                self.diffusion.p_sample_loop if not self.use_ddim else self.diffusion.ddim_sample_loop
                            )
                            y = th.tensor(([i for i in range(NUM_CLASSES)] * (run_samples_batch_size // NUM_CLASSES)), device=dist_util.dev())
                            model_kwargs = {'y': y}
                            generated_samples = sample_fn(
                                self.ddp_model,
                                (run_samples_batch_size, 3, self.image_size, self.image_size),
                                clip_denoised=self.clip_denoised,
                                model_kwargs=model_kwargs,
                            )
                            # Normalize samples to [0, 255] and change to uint8
                            samples = ((generated_samples + 1) * 127.5).clamp(0, 255).to(th.uint8)
                            # Rearrange the tensor to be in HWC format for image saving
                            samples = samples.permute(0, 2, 3, 1).contiguous()
                            if fid_samples == None:
                                fid_samples = samples
                            else:
                                fid_samples = th.cat((fid_samples, samples), dim=0)
                            if i==0:
                                samples = samples[:self.num_samples_visualize].cpu().numpy()
                                image_list = [sample for sample in samples]
                                collage = self._create_image_collage(image_list, int(np.sqrt(self.num_samples_visualize)), int(np.sqrt(self.num_samples_visualize)))
                                wandb_log_images["Sampled Images"] = wandb.Image(collage, caption="Sampled Images")
                        print(f"Completed Sampling {run_num_samples} samples!")

                        if (self.step % self.fid_interval == 0):
                            # Compute FID Score for the generated images
                            print("FID samples shape:", fid_samples.shape)
                            if self.training_data_inception_mu_sigma_path != "":
                                print("Calculating FID Score!")
                                generated_inception_mu, generated_inception_sigma = calculate_activation_statistics(fid_samples.cpu().numpy(), self.inception_model, batch_size=128, device=dist_util.dev())
                                fid_value = calculate_frechet_distance(self.training_data_inception_mu, self.training_data_inception_sigma, generated_inception_mu, generated_inception_sigma)
                                # fid_value = calculate_frechet_distance(self.training_data_inception_mu, self.training_data_inception_sigma, self.training_data_inception_mu, self.training_data_inception_sigma)
                                wandb_log_images["FID"] = fid_value
                                print(f"Calculated FID Score - {fid_value}!")
                        
                        # Plot Gaussians at the last time step for all classes if Clustered Model
                        if isinstance(self.ddp_model.module, ClusteredModel):
                            y = th.arange(NUM_CLASSES, device=dist_util.dev()).repeat(10 + 1)
                            plot_t = th.cat((th.arange(start = 0, end = self.diffusion.num_timesteps, step = self.diffusion.num_timesteps / 10, device = dist_util.dev()), th.tensor([self.diffusion.num_timesteps - 1], device=dist_util.dev()))).repeat_interleave(NUM_CLASSES).long()
                            # y = th.arange(NUM_CLASSES, device=dist_util.dev())
                            # plot_t = th.tensor([self.diffusion.num_timesteps - 1] * NUM_CLASSES, device=dist_util.dev())
                            model_kwargs = {"y": y}

                            mu_bar, sigma_bar = self.ddp_model.module.guidance_model(plot_t, self.diffusion.sqrt_one_minus_alphas_cumprod, y)
                            gaussian_image = self._plot_multiple_gaussian_contours(mu_bar, sigma_bar, plot_t)
                            wandb_log_images["Gaussian 2D Plots"] = wandb.Image(gaussian_image, caption = "Gaussian 2D Plot")

                            # Confusion Matrices for cosine similarity and distances at last time step
                            mu_bar_last_t = mu_bar[-10:]
                            mu_bar_last_t = mu_bar_last_t.view(mu_bar_last_t.shape[0], -1)
                            mu_bar_last_t = th.cat((th.zeros((1, mu_bar_last_t.shape[-1]), device=mu_bar_last_t.device), mu_bar_last_t), dim=0)

                            norm_tensor = F.normalize(mu_bar_last_t, p=2, dim=1)
                            mu_bar_last_t_cosine_similarity_matrix = th.mm(norm_tensor, norm_tensor.t())
                            mu_bar_last_t_distance = th.cdist(mu_bar_last_t, mu_bar_last_t, p=2)

                            # Creating the plot
                            fig, ax = plt.subplots(figsize=(10, 8))
                            cax = ax.matshow(mu_bar_last_t_cosine_similarity_matrix.numpy(), cmap='Blues')
                            # Adding color bar
                            plt.colorbar(cax)
                            # Adding titles and labels
                            plt.title('Distance Matrix', pad=20)
                            plt.xlabel('Classes')
                            plt.ylabel('Classes')
                            ax.set_xticks(np.arange(0, NUM_CLASSES + 1))
                            ax.set_yticks(np.arange(0, NUM_CLASSES + 1))
                            ax.set_xticklabels(np.arange(-1, NUM_CLASSES))
                            ax.set_yticklabels(np.arange(-1, NUM_CLASSES))
                            # Displaying values in each cell
                            for (i, j), val in np.ndenumerate(mu_bar_last_t_cosine_similarity_matrix):
                                ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='black')
                            wandb_log_images[f"Cosine Similarity at t={self.diffusion.num_timesteps}"] = wandb.Image(plt)
                            plt.clf()

                            # Creating the plot
                            fig, ax = plt.subplots(figsize=(10, 8))
                            cax = ax.matshow(mu_bar_last_t_distance.numpy(), cmap='Blues')
                            # Adding color bar
                            plt.colorbar(cax)
                            # Adding titles and labels
                            plt.title('Distance Matrix', pad=20)
                            plt.xlabel('Classes')
                            plt.ylabel('Classes')
                            ax.set_xticks(np.arange(0, NUM_CLASSES + 1))
                            ax.set_yticks(np.arange(0, NUM_CLASSES + 1))
                            ax.set_xticklabels(np.arange(-1, NUM_CLASSES))
                            ax.set_yticklabels(np.arange(-1, NUM_CLASSES))
                            # Displaying values in each cell
                            for (i, j), val in np.ndenumerate(mu_bar_last_t_distance):
                                ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='black')
                            wandb_log_images[f"Distance at t={self.diffusion.num_timesteps}"] = wandb.Image(plt)
                            plt.clf()

                        if self.use_wandb:
                            wandb.log(wandb_log_images, step=self.step)
                        
                    self.ddp_model.train()
                
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return

            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]

            if isinstance(self.schedule_sampler, UniformSampler) or isinstance(self.schedule_sampler, LossSecondMomentResampler):
                guidance_model_freeze = True
                denoise_model_freeze = False
                guidance_loss_freeze = True
                denoise_loss_freeze = False
                sample_condition = "not-same"
            elif isinstance(self.schedule_sampler, SameTSampler):
                if self.step >= self.no_guidance_step:
                    if self.freeze_guidance_after_no_guidance_step:
                        guidance_model_freeze = True
                    else:
                        guidance_model_freeze = False
                    denoise_model_freeze = False
                    guidance_loss_freeze = True
                    denoise_loss_freeze = False
                    sample_condition = "not-same"
                else:
                    guidance_model_freeze = False
                    denoise_model_freeze = False
                    guidance_loss_freeze = False
                    denoise_loss_freeze = False
                    sample_condition = "same"
            elif isinstance(self.schedule_sampler, AlternateSampler):
                if self.step >= self.no_guidance_step:
                    if self.freeze_guidance_after_no_guidance_step:
                        guidance_model_freeze = True
                    else:
                        guidance_model_freeze = False
                    denoise_model_freeze = False
                    guidance_loss_freeze = True
                    denoise_loss_freeze = False
                    sample_condition = "not-same"
                else:
                    if self.step % 2:
                        guidance_model_freeze = True
                        denoise_model_freeze = False
                        guidance_loss_freeze = True
                        denoise_loss_freeze = False
                        sample_condition = "not-same"
                    else:
                        guidance_model_freeze = False
                        denoise_model_freeze = False
                        guidance_loss_freeze = False
                        denoise_loss_freeze = False
                        sample_condition = "same"
            elif isinstance(self.schedule_sampler, GANTypeSampler):
                if self.step >= self.no_guidance_step:
                    if self.freeze_guidance_after_no_guidance_step:
                        guidance_model_freeze = True
                    else:
                        guidance_model_freeze = False
                    denoise_model_freeze = False
                    guidance_loss_freeze = True
                    denoise_loss_freeze = False
                    sample_condition = "not-same"
                else:
                    if self.step % 2:
                        guidance_model_freeze = True
                        denoise_model_freeze = False
                        guidance_loss_freeze = True
                        denoise_loss_freeze = False
                        sample_condition = "not-same"
                    else:
                        guidance_model_freeze = False
                        denoise_model_freeze = True
                        guidance_loss_freeze = False
                        denoise_loss_freeze = False
                        sample_condition = "same"
            
            loss_flags = {
                "guidance_model_freeze": guidance_model_freeze,
                "denoise_model_freeze": denoise_model_freeze,
                "guidance_loss_freeze": guidance_loss_freeze,
                "denoise_loss_freeze": denoise_loss_freeze,
                "sample_condition": sample_condition,
            }

            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev(), loss_flags=loss_flags)

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
                loss_flags=loss_flags
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            # THIS IS FOR BASELINE
            if isinstance(self.schedule_sampler, LossSecondMomentResampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            logged_data = log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            if dist.get_rank() == 0:
                step_values = {
                    "step": self.step + self.resume_step,
                    "samples": (self.step + self.resume_step + 1) * self.global_batch
                }
                if self.use_wandb:
                    wandb_log_data = {**step_values, **logged_data}
                    wandb_log_data["t"] = t.mean(dtype = th.float)
                    wandb.log(wandb_log_data, step=self.step)
            self.mp_trainer.backward(loss)

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    logged_data = {}
    for key, values in losses.items():
        mean_value = values.mean().item()
        logger.logkv_mean(key, mean_value)
        logged_data[key] = mean_value
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)

    return logged_data
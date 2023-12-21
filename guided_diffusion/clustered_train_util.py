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


class ClusteredTrainLoop:
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

        self.guidance_opt = AdamW(
            list(self.mp_trainer.model.guidance_model.parameters()), lr=self.lr, weight_decay=self.weight_decay
        )
        self.denoise_opt = AdamW(
            list(self.mp_trainer.model.denoise_model.parameters()), lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.guidance_ema_params, self.denoise_ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.guidance_ema_params = [
                copy.deepcopy(list(self.mp_trainer.model.guidance_model.parameters()))
                for _ in range(len(self.ema_rate))
            ]
            self.denoise_ema_params = [
                copy.deepcopy(list(self.mp_trainer.model.denoise_model.parameters()))
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

        self.loss_flags = {
            "guidance_model_freeze": False,
            "denoise_model_freeze": False,
            "guidance_loss_freeze": False,
            "denoise_loss_freeze": False,
            "sample_condition": False,
        }

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
        guidance_ema_params = copy.deepcopy(list(self.mp_trainer.model.guidance_model.parameters()))
        denoise_ema_params = copy.deepcopy(list(self.mp_trainer.model.denoise_model.parameters()))

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        guidance_ema_checkpoint = find_guidance_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        denoise_ema_checkpoint = find_denoise_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if guidance_ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {guidance_ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    guidance_ema_checkpoint, map_location=dist_util.dev()
                )
                guidance_ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)
        if denoise_ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {denoise_ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    denoise_ema_checkpoint, map_location=dist_util.dev()
                )
                denoise_ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(guidance_ema_params)
        dist_util.sync_params(denoise_ema_params)
        return guidance_ema_params, denoise_ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        guidance_opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"guidance_opt{self.resume_step:06}.pt"
        )
        denoise_opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"denoise_opt{self.resume_step:06}.pt"
        )
        if bf.exists(guidance_opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {guidance_opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                guidance_opt_checkpoint, map_location=dist_util.dev()
            )
            self.guidance_opt.load_state_dict(state_dict)
        if bf.exists(denoise_opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {denoise_opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                denoise_opt_checkpoint, map_location=dist_util.dev()
            )
            self.denoise_opt.load_state_dict(state_dict)
    
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

            if self.step < 100 or self.step % self.save_interval in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
            # if self.step % self.save_interval == 0:
                if self.step % self.save_interval == 0:
                    self.save()

                # Sample num_samples_visualize images everytime we save the model
                if dist.get_rank() == 0:  # Make sure only the master process does the sampling
                    self.ddp_model.eval()
                    wandb_log_images = {}

                    with th.no_grad():
                        if self.step % self.save_interval == 0:
                            # Generate samples
                            fid_samples = None
                            if self.step == 0:
                                run_num_samples = self.num_samples_visualize
                                run_samples_batch_size = self.num_samples_visualize
                            else:
                                run_num_samples = self.num_samples if (self.step % self.fid_interval == 0) else self.num_samples_visualize
                                run_samples_batch_size = self.num_samples_batch_size if (self.step % self.fid_interval == 0) else self.num_samples_visualize

                            for i in range(0, run_num_samples // run_samples_batch_size):
                                print(f"Sampling {run_num_samples} Images for batch number {i+1} out of {run_num_samples // self.num_samples_batch_size} batches!")
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

                            fig_cosine, axes_cosine = plt.subplots(1, 11, figsize=(55, 7))  # 11 subplots in a row for cosine similarity
                            fig_distance, axes_distance = plt.subplots(1, 11, figsize=(55, 7))  # 11 subplots in a row for distance

                            for i in range(0, mu_bar.shape[0], NUM_CLASSES):
                                t_ = plot_t[i].item()
                                mu_bar_t = mu_bar[i:i + NUM_CLASSES]
                                mu_bar_t = mu_bar_t.view(mu_bar_t.shape[0], -1)
                                mu_bar_t = th.cat((th.zeros((1, mu_bar_t.shape[-1]), device=mu_bar_t.device), mu_bar_t), dim=0)

                                norm_tensor = F.normalize(mu_bar_t, p=2, dim=1)
                                mu_bar_t_cosine_similarity_matrix = th.mm(norm_tensor, norm_tensor.t())
                                mu_bar_t_distance = th.cdist(mu_bar_t, mu_bar_t, p=2)

                                # Plotting cosine similarity matrix
                                ax = axes_cosine[i // NUM_CLASSES]
                                mu_bar_t_cosine_similarity_matrix_numpy = mu_bar_t_cosine_similarity_matrix.cpu().numpy()
                                cax1 = ax.matshow(mu_bar_t_cosine_similarity_matrix_numpy, cmap='Blues')
                                ax.set_title(f'Cosine Similarity at t={t_}')
                                ax.set_xticks(np.arange(0, NUM_CLASSES + 1))
                                ax.set_yticks(np.arange(0, NUM_CLASSES + 1))
                                ax.set_xticklabels(np.arange(-1, NUM_CLASSES))
                                ax.set_yticklabels(np.arange(-1, NUM_CLASSES))
                                for (i_, j_), val in np.ndenumerate(mu_bar_t_cosine_similarity_matrix_numpy):
                                    ax.text(j_, i_, f'{val:.2f}', ha='center', va='center', color='black')

                                # Plotting distance matrix
                                ax = axes_distance[i // NUM_CLASSES]
                                mu_bar_t_distance_numpy = mu_bar_t_distance.cpu().numpy()
                                cax2 = ax.matshow(mu_bar_t_distance_numpy, cmap='Blues')
                                ax.set_title(f'Distance at t={t_}')
                                ax.set_xticks(np.arange(0, NUM_CLASSES + 1))
                                ax.set_yticks(np.arange(0, NUM_CLASSES + 1))
                                ax.set_xticklabels(np.arange(-1, NUM_CLASSES))
                                ax.set_yticklabels(np.arange(-1, NUM_CLASSES))
                                for (i_, j_), val in np.ndenumerate(mu_bar_t_distance_numpy):
                                    ax.text(j_, i_, f'{val:.2f}', ha='center', va='center', color='black')

                            # Adding a color bar for the last subplot of each figure
                            # plt.colorbar(cax1, ax=axes_cosine[-1], orientation='vertical')
                            # plt.colorbar(cax2, ax=axes_distance[-1], orientation='vertical')

                            # Saving or displaying the figures
                            fig_cosine.suptitle('Cosine Similarity Matrices for Different Timesteps')
                            fig_distance.suptitle('Distance Matrices for Different Timesteps')

                            # Use wandb to log these images if required
                            wandb_log_images["Cosine Similarity Matrices"] = wandb.Image(fig_cosine)
                            wandb_log_images["Distance Matrices"] = wandb.Image(fig_distance)

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
        # print("STEP: ", self.step)
        # self.ddp_model.module.guidance_model.print_mu_diff("loop start")
        # self.ddp_model.module.guidance_model.print_grads("loop start")
        self.forward_backward(batch, cond)

        # self.ddp_model.module.guidance_model.print_mu_diff("before optimize")
        # self.ddp_model.module.guidance_model.print_grads("before optimize")

        guidance_took_step = False
        denoise_took_step = False

        if self.loss_flags["guidance_model_freeze"]:
            denoise_took_step = self.mp_trainer.optimize(self.denoise_opt)
        elif self.loss_flags["denoise_model_freeze"]:
            guidance_took_step = self.mp_trainer.optimize(self.guidance_opt)
        else:
            denoise_took_step = self.mp_trainer.optimize(self.denoise_opt)
            guidance_took_step = self.mp_trainer.optimize(self.guidance_opt)
        
        # self.ddp_model.module.guidance_model.print_mu_diff("after optimize")
        # self.ddp_model.module.guidance_model.print_grads("after optimize")
        if guidance_took_step:
            self._update_guidance_ema()
        if denoise_took_step:
            self._update_denoise_ema()
        # self.ddp_model.module.guidance_model.print_mu_diff("after ema")
        # self.ddp_model.module.guidance_model.print_grads("after ema")
        if guidance_took_step:
            self._anneal_guidance_lr()
        if denoise_took_step:
            self._anneal_denoise_lr()
        self.log_step()
        # self.ddp_model.module.guidance_model.print_mu_diff("loop end")
        # self.ddp_model.module.guidance_model.print_grads("loop end")

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
            else:
                raise ValueError(f"Not valid sampler!")
            
            self.loss_flags = {
                "guidance_model_freeze": guidance_model_freeze,
                "denoise_model_freeze": denoise_model_freeze,
                "guidance_loss_freeze": guidance_loss_freeze,
                "denoise_loss_freeze": denoise_loss_freeze,
                "sample_condition": sample_condition,
            }
            print(self.loss_flags)

            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev(), loss_flags=self.loss_flags)

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
                loss_flags=self.loss_flags
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

    def _update_guidance_ema(self):
        for rate, params in zip(self.ema_rate, self.guidance_ema_params):
            update_ema(params, list(self.mp_trainer.model.guidance_model.parameters()), rate=rate)
        
    def _update_denoise_ema(self):
        for rate, params in zip(self.ema_rate, self.denoise_ema_params):
            update_ema(params, list(self.mp_trainer.model.denoise_model.parameters()), rate=rate)

    def _anneal_guidance_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.guidance_opt.param_groups:
            param_group["lr"] = lr
    
    def _anneal_denoise_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.denoise_opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint(rate, params, type_="model"):

            if type_ == "model":
                state_dict = self.mp_trainer.master_params_to_state_dict(params)
            elif type_ == "guidance":
                state_dict = self.mp_trainer.guidance_params_to_state_dict(params)
            elif type_ == "denoise":
                state_dict = self.mp_trainer.denoise_params_to_state_dict(params)
            else:
                raise ValueError("INVALID ENTRY HERE. WHAT ARE YOU DOING?")
                
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if type_ == "model":
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                elif type_ == "guidance":
                    filename = f"guidance_ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                elif type_ == "denoise":
                    filename = f"denoise_ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                else:
                    raise ValueError("INVALID ENTRY HERE. WHAT ARE YOU DOING?")
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params, type_ = "model")
        for rate, params in zip(self.ema_rate, self.guidance_ema_params):
            save_checkpoint(rate, params, type_="guidance")
        for rate, params in zip(self.ema_rate, self.denoise_ema_params):
            save_checkpoint(rate, params, type_="denoise")

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"guidance_opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.guidance_opt.state_dict(), f)
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"denoise_opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.denoise_opt.state_dict(), f)

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


def find_guidance_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"guidance_ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def find_denoise_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"denoise_ema_{rate}_{(step):06d}.pt"
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
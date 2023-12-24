"""
Train a diffusion model on images.
"""

import argparse
import wandb

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()
    if args.use_wandb:
        wandb.init(
            entity = "nyu_chanukya",
            project = "Clustered_Diffusion",
            config = args,
            name = args.wandb_run_name,
        )

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        random_crop=args.random_crop,
        random_flip=args.random_flip,
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        fid_interval=args.fid_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        no_guidance_step=args.no_guidance_step,
        freeze_guidance_after_no_guidance_step=args.freeze_guidance_after_no_guidance_step,
        # Sampling arguments for visualization during training
        clip_denoised=args.clip_denoised,
        num_samples=args.num_samples,
        num_samples_batch_size=args.num_samples_batch_size,
        num_samples_visualize=args.num_samples_visualize,
        use_ddim=args.use_ddim,
        image_size=args.image_size,
        training_data_inception_mu_sigma_path=args.training_data_inception_mu_sigma_path,
        use_wandb=args.use_wandb,
    ).run_loop()

    if args.use_wandb:
        wandb.finish()


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        no_guidance_step=200000000,
        freeze_guidance_after_no_guidance_step=True,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=1000,
        fid_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        random_crop=False,
        random_flip=False,
        training_data_inception_mu_sigma_path="",
        use_wandb=True,
        wandb_run_name="wandb_run_name",
    )
    # Sampling arguments for visualization during training
    defaults.update(dict(
        clip_denoised=True,
        num_samples=400,
        num_samples_batch_size=200,
        num_samples_visualize=100,
        use_ddim=False,
    ))
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()

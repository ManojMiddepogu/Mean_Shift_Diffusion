"""
Train a diffusion model on images.
"""

import argparse
import wandb

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    clustered_model_and_diffusion_defaults,
    create_clustered_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()
    wandb.init(
		entity = "llvm",
		config = args,
	)

    logger.log("creating model and diffusion...")
    model, diffusion = create_clustered_model_and_diffusion(
        **args_to_dict(args, clustered_model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=True, # CHECK - THIS IS JUST TO LOAD THE LABELS
        # class_cond=args.class_cond,
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
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        # Sampling arguments for visualization during training
        clip_denoised=args.clip_denoised,
        num_samples_visualize=args.num_samples_visualize,
        use_ddim=args.use_ddim,
        image_size=args.image_size,
    ).run_loop()

    wandb.finish()


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        random_crop=False,
        random_flip=False,
    )
    # Sampling arguments for visualization during training
    defaults.update(dict(
        clip_denoised=True,
        num_samples_visualize=25,
        use_ddim=False,
    ))
    defaults.update(clustered_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
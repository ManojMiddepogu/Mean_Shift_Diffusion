import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch as th
import torch.distributed as dist

from guided_diffusion.nn import mean_flat
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    clustered_model_and_diffusion_defaults,
    create_clustered_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

def plot_multiple_gaussian_contours(means, sigmas):
    # Convert inputs to numpy arrays if they are not already
    means = np.array(means.to('cpu'))
    sigmas = np.array(sigmas.to('cpu'))

    # Check if the lengths of means and sigmas are equal
    if means.shape != sigmas.shape:
        raise ValueError("The shapes of means and sigmas must be equal.")

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot circles for each Gaussian distribution
    for mean, sigma in zip(means.flatten(), sigmas.flatten()):
        circle = plt.Circle((mean, 0), sigma, color='b', fill=False)
        ax.add_artist(circle)

    # Setting the limits of the plot
    ax.set_xlim(np.min(means) - 3*np.max(sigmas), np.max(means) + 3*np.max(sigmas))
    ax.set_ylim(-3*np.max(sigmas), 3*np.max(sigmas))

    # Add grid, labels and title
    ax.grid(True)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_title('Contours of Multiple 1D Gaussian Distributions')

    save_path = os.path.join(logger.get_dir(), f"gaussian_plot.png")
    plt.savefig(save_path, bbox_inches='tight')

    # plt.show()


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_clustered_model_and_diffusion(
        **args_to_dict(args, clustered_model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("gathering mu, sigma...")
    y = th.arange(NUM_CLASSES, device=dist_util.dev())
    t = th.tensor([diffusion.num_timesteps - 1] * NUM_CLASSES, device=dist_util.dev())
    model_kwargs = {"y": y}

    with th.no_grad():
        mu_bar, sigma_bar = model.guidance_model(t, diffusion.sqrt_one_minus_alphas_cumprod, y)
        mu_bar = mean_flat(mu_bar)
        print(mu_bar)
        print(sigma_bar)

        plot_multiple_gaussian_contours(mu_bar, sigma_bar)


def create_argparser():
    defaults = dict(
        model_path="",
    )
    defaults.update(clustered_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch as th
import torch.distributed as dist

from guided_diffusion.nn import mean_flat, mean_flat_2
from guided_diffusion.losses import normal_js
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    clustered_model_and_diffusion_defaults,
    create_clustered_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse

def plot_multiple_gaussian_contours(means, sigmas):
    # Convert inputs to numpy arrays if they are not already
    means = np.array(means.view(10, -1).to('cpu'))
    sigmas = np.array(sigmas.view(10, -1).to('cpu'))

    print(means.shape)
    print(sigmas.shape)

    # Check if the lengths of means and sigmas are equal
    # if means.shape != sigmas.shape:
    #     raise ValueError("The shapes of means and sigmas must be equal.")

    reduced_means = PCA(n_components=2).fit_transform(means)
    print(reduced_means)

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot circles for each Gaussian distribution
    for mean, sigma in zip(reduced_means, sigmas):
        ellipse = Ellipse(xy=mean, width=3 * sigma, height=3 * sigma, edgecolor='r', fc='None', lw=2)
        ax.add_patch(ellipse)
        ax.scatter(mean[0], mean[1], c='red', marker='x')
        # circle = plt.Circle((mean[0], mean[1]), 3, color='b', fill=False)
        # ax.add_artist(circle)

    # Setting the limits of the plot
    ax.set_xlim(np.min(reduced_means) - 3*np.max(sigmas), np.max(reduced_means) + 3*np.max(sigmas))
    ax.set_ylim(np.min(reduced_means) - 3*np.max(sigmas), np.max(reduced_means) + 3*np.max(sigmas))

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
        # mu_bar = mean_flat(mu_bar)
        print(mu_bar.shape)
        print(sigma_bar.shape)

        q_mean = mu_bar
        b, *shape = q_mean.shape
        q_log_variance = _broadcast_tensor(th.log(sigma_bar), q_mean.shape)

        q_mean_1 = q_mean.unsqueeze(1).expand(b, b, *shape)
        q_log_variance_1 = q_log_variance.unsqueeze(1).expand(b, b, *shape)
        q_mean_2 = q_mean.unsqueeze(0).expand(b, b, *shape)
        q_log_variance_2 = q_log_variance.unsqueeze(0).expand(b, b, *shape)

        distance_matrix = mean_flat_2(normal_js(q_mean_1, q_log_variance_1, q_mean_2, q_log_variance_2))
        print(distance_matrix)

        plot_multiple_gaussian_contours(mu_bar, sigma_bar)


def create_argparser():
    defaults = dict(
        model_path="",
    )
    defaults.update(clustered_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

def _broadcast_tensor(t, broadcast_shape):
    while len(t.shape) < len(broadcast_shape):
        t = t[..., None]
    return t.expand(broadcast_shape)


if __name__ == "__main__":
    main()

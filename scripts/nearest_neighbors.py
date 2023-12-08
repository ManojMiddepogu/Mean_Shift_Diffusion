import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch as th
import torch.distributed as dist

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    clustered_model_and_diffusion_defaults,
    create_clustered_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

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

    if args.samples_npz_path:
        logger.log("loading sampled images...")
        data = np.load(args.samples_npz_path)
        if args.class_cond:
            arr, label_arr = data["arr_0"], data["arr_1"]
        else:
            arr = data["arr_0"]
    else:
        logger.log("sampling...")
        all_images = []
        all_labels = []
        while len(all_images) * args.batch_size < args.num_samples:
            model_kwargs = {}
            if args.class_cond:
                classes = th.randint(low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev())
                # classes = th.tensor([0] * args.batch_size, device=dist_util.dev())
                # classes = th.tensor(([i for i in range(NUM_CLASSES)] * (args.batch_size // NUM_CLASSES)), device=dist_util.dev())
                model_kwargs["y"] = classes
            sample_fn = (
                diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
            )
            # noise = th.randn((args.batch_size, 3, args.image_size, args.image_size), device=dist_util.dev())
            sample = sample_fn(
                model,
                (args.batch_size, 3, args.image_size, args.image_size),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                # noise = noise,
            )
            sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
            sample = sample.permute(0, 2, 3, 1)
            sample = sample.contiguous()

            gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
            all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
            if args.class_cond:
                gathered_labels = [
                    th.zeros_like(classes) for _ in range(dist.get_world_size())
                ]
                dist.all_gather(gathered_labels, classes)
                all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
            logger.log(f"created {len(all_images) * args.batch_size} samples")

        arr = np.concatenate(all_images, axis=0)
        arr = arr[: args.num_samples]
        if args.class_cond:
            label_arr = np.concatenate(all_labels, axis=0)
            label_arr = label_arr[: args.num_samples]

    if dist.get_rank() == 0:
        if args.samples_npz_path == "":
            shape_str = "x".join([str(x) for x in arr.shape])
            out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
            logger.log(f"saving to {out_path}")
            if args.class_cond:
                np.savez(out_path, arr, label_arr)
            else:
                np.savez(out_path, arr)
        
        if args.save_images:
            logger.log("creating sample image file")
            
            save_images(arr[:min(args.plot_samples, args.num_samples)], shape_str)

            logger.log("sample image file complete")
        
        # arr = np.transpose(arr, [0, 3, 1, 2])

        # K-Nearest Neighbors
        mnist_images, mnist_labels = load_mnist_data(args.mnist_npz_path)
        flat_mnist_images = mnist_images.reshape(mnist_images.shape[0], -1)  # Flatten MNIST images

        # Flatten generated samples for k-NN
        flat_samples = arr.reshape(arr.shape[0], -1)

        # Compute k-NN
        nearest_neighbors = compute_nearest_neighbors(flat_samples, flat_mnist_images, k=5)
        nearest_neighbors = nearest_neighbors.reshape(nearest_neighbors.shape[0], nearest_neighbors.shape[1], args.image_size, args.image_size, 3)
        print(arr.shape)
        print(mnist_images.shape)
        print(nearest_neighbors.shape)

        save_nearest_neighbors(arr, nearest_neighbors)

    dist.barrier()
    logger.log("sampling complete")

def load_mnist_data(mnist_path):
    """
    Load MNIST data from the given path.
    """
    data = np.load(mnist_path)
    images = data['arr_0']  # Replace 'images' with the correct key if different
    # images = np.transpose(images, [0, 3, 1, 2])
    labels = data['label_arr_0']  # Replace 'labels' with the correct key if different
    return images, labels


def pairwise_distances(x, y):
    x_norm = np.sum(x**2, axis=1).reshape(-1, 1)
    y_norm = np.sum(y**2, axis=1).reshape(1, -1)
    dist = x_norm + y_norm - 2.0 * np.dot(x, y.T)
    return dist


def compute_nearest_neighbors(samples, mnist_images, k=5):
    """
    Compute the k-nearest neighbors for each sample in samples using the MNIST images.
    """
    distances = pairwise_distances(samples, mnist_images)
    sorted_row_indices = np.argsort(distances.min(axis=1))
    distances = distances[sorted_row_indices]

    neighbor_distances = np.partition(distances, k-1, axis=1)[:, :k]
    neighbor_indices = np.argpartition(distances, k-1, axis=1)[:, :k]

    nearest_neighbors = mnist_images[neighbor_indices]

    print(neighbor_indices)
    print(neighbor_distances)
    
    return nearest_neighbors


def save_nearest_neighbors(arr, neighbors):
    samples_count = arr.shape[0]
    num_rows = samples_count  # Number of rows in the grid
    num_cols = 1 + neighbors.shape[1]  # Number of columns in the grid
    num_images = num_rows * num_cols

    # Create a figure for the grid
    # plt.figure(figsize=(2*num_rows, num_cols))
    plt.figure(figsize=(10, 10))

    counter = 0
    for i in range(num_rows):
        for j in range(num_cols):
            counter += 1
            plt.subplot(num_rows, num_cols, counter)
            if j == 0:
                plt.imshow(arr[i])
            else:
                plt.imshow(neighbors[i, j-1])
            plt.axis('off')
    
    save_path = os.path.join(logger.get_dir(), f"nearest_neighbors.png")
    plt.savefig(save_path, bbox_inches='tight')

def save_images(arr, shape_str):
    samples_count = arr.shape[0]
    num_rows = int(np.sqrt(samples_count))  # Number of rows in the grid
    num_cols = num_rows  # Number of columns in the grid
    num_images = num_rows * num_cols

    # Create a figure for the grid
    # plt.figure(figsize=(2*num_rows, num_cols))
    plt.figure(figsize=(10, 10))

    for i in range(num_images):
        # Add a subplot for each image
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(arr[i])
        plt.axis('off')

    # Adjust spacing between subplots
    # plt.subplots_adjust(wspace=1.5 / num_rows, hspace=1.5 / num_cols)

    save_path = os.path.join(logger.get_dir(), f"samples_{shape_str}_{samples_count}.png")
    plt.savefig(save_path, bbox_inches='tight')


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        plot_samples=100,
        batch_size=16,
        use_ddim=False,
        model_path="",
        save_images=False,
        mnist_npz_path="",
        samples_npz_path=""
    )
    defaults.update(clustered_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()

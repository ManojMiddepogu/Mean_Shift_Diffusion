import numpy as np
import matplotlib.pyplot as plt

def visualize_images(npz_file_path):
    # Load the .npz file
    data = np.load(npz_file_path)

    # Extract the images from the loaded data
    images = data['arr_0']

    num_rows = 5  # Number of rows in the grid
    num_cols = 5  # Number of columns in the grid
    num_images = num_rows * num_cols

    # Create a figure for the grid
    plt.figure(figsize=(10, 10))

    for i in range(num_images):
        # Add a subplot for each image
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(images[i])
        plt.axis('off')
        # plt.title(f"Image {i + 1}")

    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    # Show the grid of images
    plt.savefig('/scratch/mm12799/Clustered_Diffusion/guided_diffusion/logs/logs_test_run/5x5.png')

if __name__ == "__main__":
    npz_file_path = "/scratch/mm12799/Clustered_Diffusion/guided_diffusion/logs/logs_test_run/samples_32x32x32x3.npz"  # Replace with the actual path to your .npz file
    visualize_images(npz_file_path)

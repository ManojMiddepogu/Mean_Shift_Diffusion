import os
import torchvision
from torchvision import datasets, transforms
from PIL import Image
import torch
import numpy as np

# Set the path where you want to download CIFAR-10 dataset
data_dir = "/scratch/crg9968/datasets/cifar2/"

# Create the data directory if it doesn't exist
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Define a transform to first convert the images to tensors and then resize them
transform = transforms.Compose([
    transforms.ToTensor()
])

# Download CIFAR-10 dataset without transformations
train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)

def save_images(dataset, subfolder):
    class_dirs = {}
    for i, (img_tensor, target) in enumerate(dataset):
        if target not in [0, 1]:
            continue
        if target not in class_dirs:
            class_dir = os.path.join(data_dir, subfolder, str(target))
            os.makedirs(class_dir, exist_ok=True)
            class_dirs[target] = class_dir
        
        img_path = os.path.join(class_dirs[target], f"{target}_{i:05d}.png")
        
        # Convert the tensor to PIL image
        img = transforms.ToPILImage()(img_tensor)
        img.save(img_path)

# Save training and testing images
save_images(train_dataset, "train")
save_images(test_dataset, "test")

def generate_npz_file(dataset, npz_filename):
    images = []
    labels = []

    for i, (img_tensor, label) in enumerate(dataset):
        if label not in [0, 1]:
            continue
        # Convert tensor to numpy and append to list
        # NOTE HERE THE TRANFORM APPLIES GIVES IMAGE IN [0,1] RANGE, SO NO (SAMPLE + 1)
        sample = ((img_tensor) * 255).clamp(0, 255).to(torch.uint8)
        sample = sample.permute(1, 2, 0)
        sample = sample.contiguous()

        images.append(sample)
        labels.append(label)

    # Convert lists to numpy arrays
    images = np.array(images)
    print(images.shape)
    labels = np.array(labels)

    # Save the arrays to an NPZ file
    np.savez(npz_filename, arr_0=images, label_arr_0=labels)
    print(f"Data saved to {npz_filename}")

# Generate NPZ file for training data
generate_npz_file(train_dataset, os.path.join(data_dir, "cifar_train_reference.npz"))

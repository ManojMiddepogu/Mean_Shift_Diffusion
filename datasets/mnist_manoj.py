import os
import torchvision
from torchvision import datasets, transforms
from PIL import Image
import torch

# Set the path where you want to download MNIST dataset
data_dir = "/scratch/mm12799/datasets/mnist/"

# Create the data directory if it doesn't exist
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Define a transform to first convert the images to tensors and then resize them
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((32, 32))  # Resize to 32x32
])

# Download MNIST dataset as tensors and resize them
train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

# Function to save images into subdirectories by class
def save_images(dataset, subfolder):
    class_dirs = {}
    for i, (img_tensor, target) in enumerate(dataset):
        # Convert the tensor to a 3-channel image tensor by repeating the channels
        img_tensor = img_tensor.repeat(3, 1, 1)
        
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

import os
import shutil
import torchvision

# Set the path where you want to download CIFAR-10 dataset
data_dir = "/scratch/crg9968/datasets/cifar10/"

# Create the data directory if it doesn't exist
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Download CIFAR-10 dataset without transformations
train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True)
test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True)

# Organize images into subdirectories by class
for i, (img, target) in enumerate(train_dataset):
    class_dir = os.path.join(data_dir, str(target))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)
    img_path = os.path.join(class_dir, f"{target}_{i:05d}.png")
    img.save(img_path)

# Delete the downloaded files from torchvision
shutil.rmtree(os.path.join(data_dir, "cifar-10-batches-py"))
shutil.rmtree(os.path.join(data_dir, "cifar-10-python.tar.gz"))
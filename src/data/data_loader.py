"""
Data Loader for Brain Tumor MRI images.
Loads images, resizes them, labels them (tumor=1, healthy=0),
and splits into training and validation sets.
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Path to our data (top-level yes/no folders)
DATA_DIR = Path("data/raw/brain_mri")

# Image size our model expects (ResNet needs 224x224)
IMG_SIZE = 224


def get_transforms():
    """
    Defines how each image is processed before going into the model:
    - Convert to RGB (some MRIs are grayscale, model needs 3 channels)
    - Resize to 224x224
    - Convert to a tensor (numbers the model understands)
    - Normalize using ImageNet stats (because ResNet was trained on ImageNet)
    """
    return transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_dataloaders(batch_size: int = 8, val_split: float = 0.2):
    """
    Loads all images and splits them into training + validation sets.
    - batch_size: how many images the model sees at once
    - val_split: fraction kept for validation (0.2 = 20%)
    Returns: train_loader, val_loader, class_names
    """
    # ImageFolder automatically labels images by folder name:
    #   data/raw/brain_mri/no/   -> label 0
    #   data/raw/brain_mri/yes/  -> label 1
    full_dataset = datasets.ImageFolder(
        root=str(DATA_DIR),
        transform=get_transforms(),
    )

    class_names = full_dataset.classes  # ['no', 'yes']

    # Split into training and validation
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),  # reproducible split
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, class_names


# Quick test: run this file directly to check the data loads
if __name__ == "__main__":
    train_loader, val_loader, class_names = get_dataloaders()

    print("✅ Data loader works!")
    print("Classes        :", class_names, "(0=no tumor, 1=tumor)")
    print("Training images :", len(train_loader.dataset))
    print("Validation images:", len(val_loader.dataset))

    # Grab one batch to confirm shapes
    images, labels = next(iter(train_loader))
    print("One batch images shape:", list(images.shape), "(batch, channels, H, W)")
    print("One batch labels       :", labels.tolist())
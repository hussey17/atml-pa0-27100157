from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.models import ResNet152_Weights

from .utils import ExperimentConfig


def _subset(dataset: Dataset, size: int | None, seed: int) -> Dataset:
    if size is None or size >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:size].tolist()
    return Subset(dataset, indices)


def cifar10_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    """ImageNet normalization makes CIFAR-10 compatible with pretrained weights."""
    mean = ResNet152_Weights.DEFAULT.meta["mean"] if "mean" in ResNet152_Weights.DEFAULT.meta else (0.485, 0.456, 0.406)
    std = ResNet152_Weights.DEFAULT.meta["std"] if "std" in ResNet152_Weights.DEFAULT.meta else (0.229, 0.224, 0.225)
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return train_transform, eval_transform


def build_cifar10_loaders(config: ExperimentConfig) -> tuple[DataLoader, DataLoader]:
    train_transform, eval_transform = cifar10_transforms(config.image_size)
    root = Path(config.data_dir)
    train_set = datasets.CIFAR10(root=root, train=True, download=True, transform=train_transform)
    val_set = datasets.CIFAR10(root=root, train=False, download=True, transform=eval_transform)
    train_set = _subset(train_set, config.train_subset, config.seed)
    val_set = _subset(val_set, config.val_subset, config.seed + 1)

    common = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": config.num_workers > 0,
    }
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(train_set, shuffle=True, generator=generator, **common)
    val_loader = DataLoader(val_set, shuffle=False, **common)
    return train_loader, val_loader


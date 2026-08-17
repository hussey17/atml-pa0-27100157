from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import MNIST

from .config import VAEExperimentConfig


def _subset(dataset: Dataset, size: int | None, seed: int) -> Dataset:
    if size is None or size >= len(dataset):
        return dataset
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))[:size].tolist()
    return Subset(dataset, indices)


def build_mnist_loaders(
    config: VAEExperimentConfig,
) -> tuple[DataLoader, DataLoader]:
    transform = transforms.ToTensor()
    root = Path(config.data_dir)
    train = MNIST(root=root, train=True, transform=transform, download=True)
    validation = MNIST(root=root, train=False, transform=transform, download=True)
    train = _subset(train, config.train_subset, config.seed)
    validation = _subset(validation, config.val_subset, config.seed + 1)
    common = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": config.num_workers > 0,
    }
    train_loader = DataLoader(
        train,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
        **common,
    )
    validation_loader = DataLoader(validation, shuffle=False, **common)
    return train_loader, validation_loader

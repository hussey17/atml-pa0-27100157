from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import CIFAR10

from .config import VitExperimentConfig


class ProcessedDataset(Dataset):
    def __init__(self, dataset: Dataset, processor: object):
        self.dataset = dataset
        self.processor = processor

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image, label = self.dataset[index]
        pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"][0]
        return pixel_values, int(label)


def _subset(dataset: Dataset, size: int | None, seed: int) -> Dataset:
    if size is None or size >= len(dataset):
        return dataset
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))[:size].tolist()
    return Subset(dataset, indices)


def build_cifar10_loaders(
    config: VitExperimentConfig, processor: object
) -> tuple[DataLoader, DataLoader]:
    root = Path(config.data_dir)
    train = _subset(CIFAR10(root=root, train=True, download=True), config.train_subset, config.seed)
    val = _subset(CIFAR10(root=root, train=False, download=True), config.val_subset, config.seed + 1)
    train = ProcessedDataset(train, processor)
    val = ProcessedDataset(val, processor)
    common = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": config.num_workers > 0,
    }
    train_loader = DataLoader(train, shuffle=False, **common)
    val_loader = DataLoader(val, shuffle=False, **common)
    return train_loader, val_loader

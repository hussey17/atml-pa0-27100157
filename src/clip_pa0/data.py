from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import STL10

from .config import ClipExperimentConfig
from .prompts import STL10_CLASSES


class IndexedDataset(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        image, label = self.dataset[index]
        source_index = self.dataset.indices[index] if isinstance(self.dataset, Subset) else index
        return image, int(label), int(source_index)


def balanced_indices(labels: Sequence[int], count: int, seed: int) -> list[int]:
    """Select a reproducible, near-equal number of examples per class."""
    labels_array = np.asarray(labels)
    classes = np.unique(labels_array)
    if count > len(labels_array):
        raise ValueError(f"Requested {count} samples from a dataset of size {len(labels_array)}")
    generator = np.random.default_rng(seed)
    base, remainder = divmod(count, len(classes))
    selected: list[int] = []
    for position, class_id in enumerate(classes):
        class_indices = np.flatnonzero(labels_array == class_id)
        take = base + (position < remainder)
        if take > len(class_indices):
            raise ValueError(f"Class {class_id} does not contain {take} samples")
        selected.extend(generator.choice(class_indices, size=take, replace=False).tolist())
    generator.shuffle(selected)
    return selected


def _random_subset(dataset: Dataset, count: int | None, seed: int) -> Dataset:
    if count is None or count >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    return Subset(dataset, torch.randperm(len(dataset), generator=generator)[:count].tolist())


def build_stl10_loader(
    config: ClipExperimentConfig,
    preprocess: object,
    *,
    split: str,
    balanced_count: int | None = None,
    subset_count: int | None = None,
    seed_offset: int = 0,
) -> DataLoader:
    dataset = STL10(
        root=Path(config.data_dir),
        split=split,
        transform=preprocess,
        download=True,
    )
    if tuple(dataset.classes) != STL10_CLASSES:
        raise RuntimeError(f"Unexpected STL-10 class order: {dataset.classes}")
    if balanced_count is not None:
        dataset = Subset(
            dataset,
            balanced_indices(dataset.labels, balanced_count, config.seed + seed_offset),
        )
    else:
        dataset = _random_subset(dataset, subset_count, config.seed + seed_offset)
    return DataLoader(
        IndexedDataset(dataset),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
    )

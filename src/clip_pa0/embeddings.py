from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .prompts import l2_normalize


@dataclass
class ImageEmbeddings:
    raw: torch.Tensor
    normalized: torch.Tensor
    labels: torch.Tensor
    indices: torch.Tensor

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            raw=self.raw.numpy(),
            normalized=self.normalized.numpy(),
            labels=self.labels.numpy(),
            indices=self.indices.numpy(),
        )


@torch.inference_mode()
def extract_image_embeddings(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> ImageEmbeddings:
    raw_batches, label_batches, index_batches = [], [], []
    model.eval()
    for images, labels, indices in loader:
        features = model.encode_image(images.to(device, non_blocking=True)).float().cpu()
        raw_batches.append(features)
        label_batches.append(labels.cpu())
        index_batches.append(indices.cpu())
    if not raw_batches:
        raise ValueError("Cannot extract embeddings from an empty data loader")
    raw = torch.cat(raw_batches)
    return ImageEmbeddings(
        raw=raw,
        normalized=l2_normalize(raw),
        labels=torch.cat(label_batches).long(),
        indices=torch.cat(index_batches).long(),
    )

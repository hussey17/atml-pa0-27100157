from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch import nn
from torch.utils.data import DataLoader

from .modeling import get_module


FEATURE_LAYERS = {
    "early": "layer1",
    "middle": "layer3",
    "late": "avgpool",
}


def collect_features(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_samples: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Capture activations with hooks and global-average-pool feature maps."""
    captured: dict[str, torch.Tensor] = {}
    buckets: dict[str, list[np.ndarray]] = {name: [] for name in FEATURE_LAYERS}
    label_batches: list[np.ndarray] = []
    handles = []

    for name, path in FEATURE_LAYERS.items():
        def hook(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor, *, key: str = name) -> None:
            captured[key] = output.detach()
        handles.append(get_module(model, path).register_forward_hook(hook))

    model.eval()
    seen = 0
    try:
        with torch.inference_mode():
            for images, labels in loader:
                remaining = max_samples - seen
                if remaining <= 0:
                    break
                images = images[:remaining].to(device, non_blocking=True)
                labels = labels[:remaining]
                model(images)
                for name, tensor in captured.items():
                    if tensor.ndim == 4:
                        tensor = tensor.mean(dim=(-2, -1))
                    buckets[name].append(tensor.flatten(1).cpu().numpy())
                label_batches.append(labels.cpu().numpy())
                seen += labels.size(0)
    finally:
        for handle in handles:
            handle.remove()

    features = {name: np.concatenate(parts, axis=0) for name, parts in buckets.items()}
    labels = np.concatenate(label_batches, axis=0)
    return features, labels


def plot_tsne(
    features: dict[str, np.ndarray],
    labels: np.ndarray,
    output_path: str | Path,
    seed: int,
    perplexity: float,
) -> None:
    figure, axes = plt.subplots(1, len(features), figsize=(18, 5), constrained_layout=True)
    colors = plt.get_cmap("tab10", 10)
    for axis, (name, matrix) in zip(axes, features.items()):
        effective_perplexity = min(perplexity, max(2.0, (len(matrix) - 1) / 3))
        embedding = TSNE(
            n_components=2,
            perplexity=effective_perplexity,
            init="pca",
            learning_rate="auto",
            random_state=seed,
        ).fit_transform(matrix)
        scatter = axis.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap=colors, s=8, alpha=0.75)
        axis.set_title(f"{name.title()} features ({matrix.shape[1]}D)")
        axis.set_xticks([])
        axis.set_yticks([])
    handles, _ = scatter.legend_elements(num=10)
    figure.legend(handles, [str(index) for index in range(10)], title="CIFAR-10 class", loc="outside lower center", ncol=10)
    figure.suptitle("ResNet-152 feature hierarchy (t-SNE)")
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


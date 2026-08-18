from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import orthogonal_procrustes

from .modality import joint_tsne, modality_gap_statistics


def fit_orthogonal_alignment(
    images: np.ndarray, texts: np.ndarray
) -> tuple[np.ndarray, dict[str, float]]:
    if images.shape != texts.shape:
        raise ValueError(f"Alignment matrices must match, got {images.shape} and {texts.shape}")
    rotation, singular_value_sum = orthogonal_procrustes(images, texts)
    aligned = images @ rotation
    identity = np.eye(rotation.shape[0], dtype=rotation.dtype)
    diagnostics = {
        "singular_value_sum": float(singular_value_sum),
        "orthogonality_error_frobenius": float(np.linalg.norm(rotation.T @ rotation - identity)),
        "residual_frobenius_before": float(np.linalg.norm(images - texts)),
        "residual_frobenius_after": float(np.linalg.norm(aligned - texts)),
        "mean_paired_cosine_before": modality_gap_statistics(images, texts)[
            "mean_paired_cosine_similarity"
        ],
        "mean_paired_cosine_after": modality_gap_statistics(aligned, texts)[
            "mean_paired_cosine_similarity"
        ],
    }
    return rotation, diagnostics


def save_alignment_visualization(
    images: np.ndarray,
    aligned_images: np.ndarray,
    texts: np.ndarray,
    labels: np.ndarray,
    output_dir: str | Path,
    *,
    seed: int,
    perplexity: float,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    before_2d, after_2d, text_2d = joint_tsne(
        [images, aligned_images, texts], seed, perplexity
    )
    np.savez_compressed(
        output_dir / "alignment_tsne_coordinates.npz",
        before=before_2d,
        after=after_2d,
        text=text_2d,
        labels=labels,
    )
    figure, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    for axis, image_coordinates, title in (
        (axes[0], before_2d, "Before orthogonal alignment"),
        (axes[1], after_2d, "After orthogonal alignment"),
    ):
        color_plot = axis.scatter(
            image_coordinates[:, 0],
            image_coordinates[:, 1],
            c=labels,
            cmap="tab10",
            marker="o",
            s=28,
            alpha=0.72,
            label="image",
        )
        axis.scatter(
            text_2d[:, 0],
            text_2d[:, 1],
            c=labels,
            cmap="tab10",
            marker="x",
            s=38,
            linewidths=1.2,
            label="paired text",
        )
        axis.set(title=title, xlabel="joint t-SNE 1", ylabel="joint t-SNE 2")
        axis.grid(alpha=0.12)
        axis.legend(loc="best")
    colorbar = figure.colorbar(color_plot, ax=axes, ticks=range(10), shrink=0.85)
    colorbar.set_label("STL-10 class index")
    figure.suptitle("One shared t-SNE fit: original, aligned, and text embeddings")
    figure.savefig(output_dir / "alignment_tsne.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_alignment_metrics(metrics: dict[str, Any], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "alignment_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

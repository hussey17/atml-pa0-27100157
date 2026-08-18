from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


def _row_normalize(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), eps)


def modality_gap_statistics(images: np.ndarray, texts: np.ndarray) -> dict[str, float]:
    if images.shape != texts.shape:
        raise ValueError(
            f"Paired image/text arrays must have equal shapes, got {images.shape} and {texts.shape}"
        )
    image_unit = _row_normalize(images)
    text_unit = _row_normalize(texts)
    return {
        "mean_image_norm": float(np.linalg.norm(images, axis=1).mean()),
        "mean_text_norm": float(np.linalg.norm(texts, axis=1).mean()),
        "centroid_euclidean_distance": float(
            np.linalg.norm(images.mean(axis=0) - texts.mean(axis=0))
        ),
        "mean_paired_euclidean_distance": float(np.linalg.norm(images - texts, axis=1).mean()),
        "mean_paired_cosine_similarity": float(np.sum(image_unit * text_unit, axis=1).mean()),
    }


def joint_tsne(groups: list[np.ndarray], seed: int, perplexity: float) -> list[np.ndarray]:
    lengths = [len(group) for group in groups]
    combined = np.concatenate(groups, axis=0)
    effective_perplexity = min(float(perplexity), float(len(combined) - 1))
    coordinates = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(combined)
    split_points = np.cumsum(lengths)[:-1]
    return list(np.split(coordinates, split_points))


def _draw_modalities(
    axis: plt.Axes,
    image_coordinates: np.ndarray,
    text_coordinates: np.ndarray,
    labels: np.ndarray,
    title: str,
) -> None:
    image_plot = axis.scatter(
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
        text_coordinates[:, 0],
        text_coordinates[:, 1],
        c=labels,
        cmap="tab10",
        marker="x",
        s=38,
        linewidths=1.2,
        label="paired text",
    )
    axis.set(title=title, xlabel="t-SNE 1", ylabel="t-SNE 2")
    axis.grid(alpha=0.12)
    axis.legend(loc="best")
    return image_plot


def analyze_modality_gap(
    raw_images: np.ndarray,
    raw_texts: np.ndarray,
    labels: np.ndarray,
    output_dir: str | Path,
    *,
    seed: int,
    perplexity: float,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized_images = _row_normalize(raw_images)
    normalized_texts = _row_normalize(raw_texts)
    raw_image_2d, raw_text_2d = joint_tsne([raw_images, raw_texts], seed, perplexity)
    norm_image_2d, norm_text_2d = joint_tsne(
        [normalized_images, normalized_texts], seed, perplexity
    )
    statistics = {
        "samples": int(len(labels)),
        "raw": modality_gap_statistics(raw_images, raw_texts),
        "l2_normalized": modality_gap_statistics(normalized_images, normalized_texts),
        "interpretation_note": (
            "t-SNE preserves local neighborhoods, not absolute global distance; use the numeric "
            "paired and centroid measures alongside the plots."
        ),
    }
    with (output_dir / "modality_gap_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(statistics, handle, indent=2)
    np.savez_compressed(
        output_dir / "modality_gap_embeddings.npz",
        raw_images=raw_images,
        raw_texts=raw_texts,
        normalized_images=normalized_images,
        normalized_texts=normalized_texts,
        labels=labels,
        raw_image_tsne=raw_image_2d,
        raw_text_tsne=raw_text_2d,
        normalized_image_tsne=norm_image_2d,
        normalized_text_tsne=norm_text_2d,
    )
    figure, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    color_plot = _draw_modalities(
        axes[0], raw_image_2d, raw_text_2d, labels, "Raw encoder embeddings"
    )
    _draw_modalities(
        axes[1], norm_image_2d, norm_text_2d, labels, "L2-normalized embeddings"
    )
    colorbar = figure.colorbar(color_plot, ax=axes, ticks=range(10), shrink=0.85)
    colorbar.set_label("STL-10 class index")
    figure.suptitle("CLIP image/text modality gap on paired STL-10 samples")
    figure.savefig(output_dir / "modality_gap_tsne.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    return statistics

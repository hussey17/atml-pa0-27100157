from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix


def classification_metrics(
    image_embeddings: torch.Tensor,
    labels: torch.Tensor,
    text_prototypes: torch.Tensor,
    class_names: tuple[str, ...],
) -> dict[str, Any]:
    if image_embeddings.ndim != 2 or text_prototypes.ndim != 2:
        raise ValueError("Embeddings and prototypes must be rank-two matrices")
    if image_embeddings.shape[1] != text_prototypes.shape[1]:
        raise ValueError("Image and text embedding dimensions do not match")
    predictions = (image_embeddings @ text_prototypes.T).argmax(dim=1)
    labels_np = labels.cpu().numpy()
    predictions_np = predictions.cpu().numpy()
    matrix = confusion_matrix(labels_np, predictions_np, labels=range(len(class_names)))
    row_counts = matrix.sum(axis=1)
    class_accuracy = np.divide(
        matrix.diagonal(),
        row_counts,
        out=np.zeros(len(class_names), dtype=float),
        where=row_counts != 0,
    )
    return {
        "samples": int(len(labels)),
        "correct": int((predictions == labels).sum()),
        "accuracy": float((predictions == labels).float().mean()),
        "per_class_accuracy": {
            name: float(value) for name, value in zip(class_names, class_accuracy, strict=True)
        },
        "confusion_matrix": matrix.tolist(),
    }


def save_zero_shot_results(results: dict[str, dict[str, Any]], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    rows = [
        {
            "prompt_strategy": strategy,
            "samples": metrics["samples"],
            "correct": metrics["correct"],
            "accuracy": metrics["accuracy"],
        }
        for strategy, metrics in results.items()
    ]
    with (output_dir / "accuracy.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    figure, axis = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    axis.bar([row["prompt_strategy"] for row in rows], [row["accuracy"] for row in rows])
    axis.set(title="CLIP zero-shot accuracy on STL-10", ylabel="Top-1 accuracy", ylim=(0, 1))
    axis.grid(axis="y", alpha=0.2)
    for index, row in enumerate(rows):
        axis.text(index, row["accuracy"] + 0.015, f"{row['accuracy']:.3f}", ha="center")
    figure.savefig(output_dir / "prompt_accuracy.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

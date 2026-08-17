from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

from .utils import write_history


def _read_metrics(path: Path) -> list[dict[str, float]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def summarize_runs(output_dir: str | Path) -> tuple[Path, Path]:
    """Create a compact table and learning curves from all completed runs."""
    output_dir = Path(output_dir)
    histories: dict[str, list[dict[str, float]]] = {}
    for metrics_path in sorted(output_dir.glob("*/metrics.csv")):
        rows = _read_metrics(metrics_path)
        if rows:
            histories[metrics_path.parent.name] = rows
    if not histories:
        raise FileNotFoundError(f"No */metrics.csv files found under {output_dir}")

    summary = []
    for name, rows in histories.items():
        best = max(rows, key=lambda row: row["val_accuracy"])
        final = rows[-1]
        summary.append(
            {
                "experiment": name,
                "epochs": int(final["epoch"]),
                "best_epoch": int(best["epoch"]),
                "best_val_accuracy": best["val_accuracy"],
                "final_train_accuracy": final["train_accuracy"],
                "final_val_accuracy": final["val_accuracy"],
                "total_train_seconds": sum(row["train_seconds"] for row in rows),
            }
        )
    summary_path = output_dir / "comparison.csv"
    write_history(summary, summary_path)

    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for name, rows in histories.items():
        epochs = [row["epoch"] for row in rows]
        axes[0].plot(epochs, [row["train_loss"] for row in rows], linestyle="--", alpha=0.75)
        axes[0].plot(epochs, [row["val_loss"] for row in rows], label=name)
        axes[1].plot(epochs, [100 * row["train_accuracy"] for row in rows], linestyle="--", alpha=0.75)
        axes[1].plot(epochs, [100 * row["val_accuracy"] for row in rows], label=name)
    axes[0].set(title="Loss (solid: validation; dashed: train)", xlabel="Epoch", ylabel="Cross-entropy")
    axes[1].set(title="Accuracy (solid: validation; dashed: train)", xlabel="Epoch", ylabel="Accuracy (%)")
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.legend(loc="outside lower center", ncol=2)
    figure_path = output_dir / "training_curves.png"
    figure.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return summary_path, figure_path


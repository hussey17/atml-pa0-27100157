from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import VitExperimentConfig
from .modeling import final_tokens, pool_tokens


def patch_indices(
    grid_size: int,
    fraction: float,
    mode: str,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must lie in [0, 1]")
    patch_count = grid_size * grid_size
    count = round(patch_count * fraction)
    if mode == "random":
        return torch.randperm(patch_count, generator=generator)[:count]
    if mode == "center":
        rows, columns = torch.meshgrid(
            torch.arange(grid_size), torch.arange(grid_size), indexing="ij"
        )
        center = (grid_size - 1) / 2
        distances = (rows - center).square() + (columns - center).square()
        return distances.flatten().argsort()[:count]
    raise ValueError(f"Unknown masking mode: {mode}")


def mask_pixel_patches(
    pixel_values: torch.Tensor,
    patch_size: int,
    fraction: float,
    mode: str,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Replace selected normalized patches with zero (the processor mean color)."""
    masked = pixel_values.clone()
    height, width = masked.shape[-2:]
    if height != width or height % patch_size:
        raise ValueError("Expected a square image divisible by patch_size")
    grid_size = height // patch_size
    for image_index in range(masked.size(0)):
        for index in patch_indices(grid_size, fraction, mode, generator):
            row, column = divmod(int(index), grid_size)
            masked[
                image_index,
                :,
                row * patch_size : (row + 1) * patch_size,
                column * patch_size : (column + 1) * patch_size,
            ] = 0
    return masked


def evaluate_masking(
    model: nn.Module,
    probes: tuple[nn.Module, nn.Module],
    loader: DataLoader,
    patch_size: int,
    fraction: float,
    mode: str,
    seed: int,
    device: torch.device,
) -> tuple[float, float]:
    generator = torch.Generator().manual_seed(seed)
    correct_cls, correct_mean, count = 0, 0, 0
    model.eval()
    with torch.inference_mode():
        for pixel_values, labels in loader:
            pixel_values = mask_pixel_patches(
                pixel_values, patch_size, fraction, mode, generator
            ).to(device)
            labels = labels.to(device)
            cls_features, mean_features = pool_tokens(final_tokens(model, pixel_values))
            correct_cls += int((probes[0](cls_features).argmax(dim=-1) == labels).sum())
            correct_mean += int((probes[1](mean_features).argmax(dim=-1) == labels).sum())
            count += len(labels)
    return correct_cls / count, correct_mean / count


def run_masking_experiment(
    model: nn.Module,
    probes: tuple[nn.Module, nn.Module],
    loader: DataLoader,
    config: VitExperimentConfig,
    patch_size: int,
    device: torch.device,
    output_dir: str | Path,
) -> list[dict[str, object]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.to(device).eval()
    rows = []
    for fraction in config.mask_fractions:
        for mode in ("random", "center"):
            trials = config.random_mask_trials if mode == "random" and fraction > 0 else 1
            for trial in range(trials):
                cls_accuracy, mean_accuracy = evaluate_masking(
                    model,
                    probes,
                    loader,
                    patch_size,
                    fraction,
                    mode,
                    config.seed + trial,
                    device,
                )
                rows.append(
                    {
                        "mask_mode": mode,
                        "fraction": fraction,
                        "trial": trial,
                        "cls_accuracy": cls_accuracy,
                        "mean_accuracy": mean_accuracy,
                    }
                )
    with (output_dir / "masking_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for axis, pooling in zip(axes, ("cls", "mean")):
        for mode in ("random", "center"):
            points = []
            for fraction in config.mask_fractions:
                values = [
                    float(row[f"{pooling}_accuracy"])
                    for row in rows
                    if row["mask_mode"] == mode and row["fraction"] == fraction
                ]
                points.append(100 * sum(values) / len(values))
            axis.plot(config.mask_fractions, points, marker="o", label=mode.title())
        axis.set(
            title=f"{pooling.upper()} probe robustness",
            xlabel="Masked patch fraction",
            ylabel="CIFAR-10 accuracy (%)",
        )
        axis.grid(alpha=0.2)
        axis.legend()
    figure.savefig(output_dir / "masking_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    return rows

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as functional
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from .modeling import MNISTVAE


def collect_latent_means(
    model: MNISTVAE,
    loader: DataLoader,
    device: torch.device,
    max_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    means, labels = [], []
    seen = 0
    model.eval()
    with torch.inference_mode():
        for images, batch_labels in loader:
            remaining = max_samples - seen
            if remaining <= 0:
                break
            images = images[:remaining].to(device, non_blocking=True)
            batch_labels = batch_labels[:remaining]
            mu, _ = model.encode(images)
            means.append(mu.cpu().numpy())
            labels.append(batch_labels.numpy())
            seen += len(batch_labels)
    return np.concatenate(means), np.concatenate(labels)


def plot_latent_space(
    model: MNISTVAE,
    loader: DataLoader,
    device: torch.device,
    max_samples: int,
    output_dir: str | Path,
    seed: int,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    means, labels = collect_latent_means(model, loader, device, max_samples)
    if means.shape[1] == 1:
        embedding = np.column_stack((means[:, 0], np.zeros(len(means))))
        reduction = "one latent coordinate plus a zero display axis"
    elif means.shape[1] == 2:
        embedding = means
        reduction = "direct 2D posterior means"
    else:
        embedding = PCA(n_components=2, random_state=seed).fit_transform(means)
        reduction = "PCA of posterior means"
    np.savez_compressed(
        output_dir / "latent_embeddings.npz",
        posterior_means=means,
        embedding_2d=embedding,
        labels=labels,
    )
    figure, axis = plt.subplots(figsize=(8, 7), constrained_layout=True)
    scatter = axis.scatter(
        embedding[:, 0], embedding[:, 1], c=labels, cmap="tab10", s=7, alpha=0.65
    )
    handles, _ = scatter.legend_elements(num=10)
    axis.legend(handles, [str(index) for index in range(10)], title="Digit", ncol=2)
    axis.set(
        title=f"MNIST posterior means ({reduction})",
        xlabel="Latent coordinate 1",
        ylabel="Latent coordinate 2",
    )
    axis.grid(alpha=0.15)
    figure.savefig(output_dir / "latent_space.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    return {
        "samples": len(labels),
        "latent_dim": means.shape[1],
        "reduction": reduction,
    }


def _selected_examples(loader: DataLoader, count: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    dataset = loader.dataset
    count = min(count, len(dataset))
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))[:count]
    examples = [dataset[int(index)] for index in indices]
    return torch.stack([item[0] for item in examples]), torch.tensor([int(item[1]) for item in examples])


def reconstruction_analysis(
    model: MNISTVAE,
    loader: DataLoader,
    device: torch.device,
    sample_count: int,
    output_dir: str | Path,
    seed: int,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images, labels = _selected_examples(loader, sample_count, seed)
    with torch.inference_mode():
        mu, _ = model.encode(images.to(device))
        reconstructions = model.decode(mu).reshape(-1, 1, 28, 28).cpu()
    figure, axes = plt.subplots(
        2, len(images), figsize=(1.8 * len(images), 4), squeeze=False, constrained_layout=True
    )
    for index in range(len(images)):
        axes[0, index].imshow(images[index, 0], cmap="gray", vmin=0, vmax=1)
        axes[0, index].set_title(f"Digit {int(labels[index])}")
        axes[1, index].imshow(reconstructions[index, 0], cmap="gray", vmin=0, vmax=1)
        axes[0, index].axis("off")
        axes[1, index].axis("off")
    figure.suptitle("Deterministic reconstructions using posterior mean", fontsize=14)
    figure.savefig(output_dir / "reconstructions.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    per_class = {str(digit): {"count": 0, "bce_sum": 0.0, "mse_sum": 0.0} for digit in range(10)}
    model.eval()
    with torch.inference_mode():
        for batch_images, batch_labels in loader:
            batch_images = batch_images.to(device)
            mu, _ = model.encode(batch_images)
            logits = model.decode_logits(mu)
            probabilities = logits.sigmoid()
            flat = batch_images.flatten(1)
            bce = functional.binary_cross_entropy_with_logits(logits, flat, reduction="none").mean(dim=1)
            mse = functional.mse_loss(probabilities, flat, reduction="none").mean(dim=1)
            for digit in range(10):
                mask = batch_labels == digit
                count = int(mask.sum())
                if count:
                    per_class[str(digit)]["count"] += count
                    per_class[str(digit)]["bce_sum"] += float(bce[mask.to(device)].sum())
                    per_class[str(digit)]["mse_sum"] += float(mse[mask.to(device)].sum())
    metrics = {}
    for digit, values in per_class.items():
        count = int(values["count"])
        metrics[digit] = {
            "count": count,
            "mean_pixel_bce": values["bce_sum"] / count if count else None,
            "mean_pixel_mse": values["mse_sum"] / count if count else None,
        }
    with (output_dir / "reconstruction_metrics_by_class.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    return {"selected_labels": labels.tolist(), "per_class": metrics}


def generation_analysis(
    model: MNISTVAE,
    device: torch.device,
    sample_count: int,
    output_dir: str | Path,
    seed: int,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    latent = torch.randn(sample_count, model.latent_dim, generator=generator)
    with torch.inference_mode():
        generated = model.decode(latent.to(device)).reshape(-1, 28, 28).cpu()
    figure, axes = plt.subplots(1, sample_count, figsize=(1.8 * sample_count, 2.2), squeeze=False)
    for index, axis in enumerate(axes[0]):
        axis.imshow(generated[index], cmap="gray", vmin=0, vmax=1)
        axis.set_title(f"z{index + 1}")
        axis.axis("off")
    figure.suptitle("Samples decoded from z ~ N(0, I)")
    figure.tight_layout()
    figure.savefig(output_dir / "generated_samples.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    result = {"latent_dim": model.latent_dim, "latent_samples": latent.tolist()}
    with (output_dir / "generation_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return result


def run_analysis(
    model: MNISTVAE,
    loader: DataLoader,
    device: torch.device,
    output_dir: str | Path,
    *,
    sample_count: int,
    latent_plot_samples: int,
    seed: int,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "latent_space": plot_latent_space(
            model, loader, device, latent_plot_samples, output_dir, seed
        ),
        "reconstructions": reconstruction_analysis(
            model, loader, device, sample_count, output_dir, seed
        ),
        "generation": generation_analysis(model, device, sample_count, output_dir, seed),
    }
    with (output_dir / "analysis_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary

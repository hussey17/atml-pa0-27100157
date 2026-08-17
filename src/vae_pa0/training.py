from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from .config import VAEExperimentConfig
from .modeling import MNISTVAE, negative_elbo


def run_epoch(
    model: MNISTVAE,
    loader: DataLoader,
    device: torch.device,
    beta: float,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"elbo": 0.0, "reconstruction": 0.0, "kl": 0.0}
    sample_count = 0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            output = model(images)
            loss = negative_elbo(output.logits, images, output.mu, output.logvar, beta)
            if training:
                loss.total.backward()
                optimizer.step()
            batch_size = len(images)
            totals["elbo"] += loss.total.item() * batch_size
            totals["reconstruction"] += loss.reconstruction.item() * batch_size
            totals["kl"] += loss.kl.item() * batch_size
            sample_count += batch_size
    return {name: value / sample_count for name, value in totals.items()}


def _write_metrics(rows: list[dict[str, float | int]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_history(history: list[dict[str, float | int]], path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    epochs = [int(row["epoch"]) for row in history]
    axes[0].plot(epochs, [row["train_elbo"] for row in history], marker="o", label="Train")
    axes[0].plot(epochs, [row["val_elbo"] for row in history], marker="o", label="Validation")
    axes[0].set(title="Negative ELBO", xlabel="Epoch", ylabel="Nats per image")
    axes[1].plot(
        epochs, [row["val_reconstruction"] for row in history], marker="o", label="Reconstruction"
    )
    axes[1].plot(epochs, [row["val_kl"] for row in history], marker="o", label="KL")
    axes[1].set(title="Validation loss components", xlabel="Epoch", ylabel="Nats per image")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend()
        axis.set_xticks(epochs)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def train_vae(
    model: MNISTVAE,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    config: VAEExperimentConfig,
    device: torch.device,
    output_dir: str | Path,
    *,
    epochs: int | None = None,
) -> list[dict[str, float | int]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.to(device)
    optimizer = Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    history: list[dict[str, float | int]] = []
    best_validation_elbo = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    epoch_count = epochs if epochs is not None else config.epochs

    for epoch in range(1, epoch_count + 1):
        train_metrics = run_epoch(model, train_loader, device, config.beta, optimizer)
        validation_metrics = run_epoch(model, validation_loader, device, config.beta)
        row: dict[str, float | int] = {"epoch": epoch}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in validation_metrics.items()})
        history.append(row)
        _write_metrics(history, output_dir / "metrics.csv")
        print(
            f"epoch={epoch:02d} train_elbo={train_metrics['elbo']:.3f} "
            f"val_elbo={validation_metrics['elbo']:.3f} "
            f"val_recon={validation_metrics['reconstruction']:.3f} "
            f"val_kl={validation_metrics['kl']:.3f}"
        )
        if validation_metrics["elbo"] < best_validation_elbo:
            best_validation_elbo = validation_metrics["elbo"]
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("Training produced no checkpoint")
    checkpoint = {
        "model_state_dict": best_state,
        "input_dim": model.input_dim,
        "hidden_dim": model.hidden_dim,
        "latent_dim": model.latent_dim,
        "beta": config.beta,
        "best_validation_elbo": best_validation_elbo,
    }
    torch.save(checkpoint, output_dir / "best_model.pt")
    _plot_history(history, output_dir / "learning_curves.png")
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "architecture": model.architecture(),
                "best_validation_elbo": best_validation_elbo,
                "config": config.as_dict(),
            },
            handle,
            indent=2,
        )
    return history


def load_vae(path: str | Path, device: torch.device) -> tuple[MNISTVAE, dict[str, object]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = MNISTVAE(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        latent_dim=int(checkpoint["latent_dim"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval(), checkpoint

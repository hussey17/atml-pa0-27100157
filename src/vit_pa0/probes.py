from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from .config import VitExperimentConfig
from .modeling import final_tokens, pool_tokens


def extract_pooled_features(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model.to(device).eval()
    cls_parts, mean_parts, label_parts = [], [], []
    with torch.inference_mode():
        for pixel_values, labels in loader:
            tokens = final_tokens(model, pixel_values.to(device, non_blocking=True))
            cls_features, mean_features = pool_tokens(tokens)
            cls_parts.append(cls_features.cpu())
            mean_parts.append(mean_features.cpu())
            label_parts.append(labels.cpu())
    return torch.cat(cls_parts), torch.cat(mean_parts), torch.cat(label_parts)


def _evaluate(
    probe: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> tuple[float, float]:
    loader = DataLoader(TensorDataset(features, labels), batch_size=batch_size)
    criterion = nn.CrossEntropyLoss(reduction="sum")
    loss, correct, count = 0.0, 0, 0
    probe.eval()
    with torch.inference_mode():
        for batch_features, batch_labels in loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            logits = probe(batch_features)
            loss += criterion(logits, batch_labels).item()
            correct += int((logits.argmax(dim=-1) == batch_labels).sum())
            count += len(batch_labels)
    return loss / count, correct / count


def train_probe(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    config: VitExperimentConfig,
    device: torch.device,
) -> tuple[nn.Linear, list[dict[str, float | int]]]:
    torch.manual_seed(config.seed)
    probe = nn.Linear(train_features.size(1), 10).to(device)
    optimizer = AdamW(
        probe.parameters(), lr=config.probe_learning_rate, weight_decay=config.probe_weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(train_features, train_labels),
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    history = []
    for epoch in range(1, config.probe_epochs + 1):
        probe.train()
        loss_sum, correct, count = 0.0, 0, 0
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = probe(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(labels)
            correct += int((logits.argmax(dim=-1) == labels).sum())
            count += len(labels)
        val_loss, val_accuracy = _evaluate(probe, val_features, val_labels, config.batch_size, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss_sum / count,
                "train_accuracy": correct / count,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }
        )
    return probe.cpu(), history


def _write_history(rows: list[dict[str, object]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_probe_experiment(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: VitExperimentConfig,
    device: torch.device,
    output_dir: str | Path,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_cls, train_mean, train_labels = extract_pooled_features(model, train_loader, device)
    val_cls, val_mean, val_labels = extract_pooled_features(model, val_loader, device)
    cls_probe, cls_history = train_probe(train_cls, train_labels, val_cls, val_labels, config, device)
    mean_probe, mean_history = train_probe(train_mean, train_labels, val_mean, val_labels, config, device)

    combined = []
    for pooling, history in (("cls", cls_history), ("mean", mean_history)):
        for row in history:
            combined.append({"pooling": pooling, **row})
    _write_history(combined, output_dir / "probe_metrics.csv")
    torch.save(
        {
            "model_id": config.model_id,
            "hidden_size": train_cls.size(1),
            "num_classes": 10,
            "cls_state_dict": cls_probe.state_dict(),
            "mean_state_dict": mean_probe.state_dict(),
        },
        output_dir / "linear_probes.pt",
    )

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for name, history in (("CLS", cls_history), ("Mean patches", mean_history)):
        axes[0].plot(
            [row["epoch"] for row in history], [row["val_loss"] for row in history], label=name
        )
        axes[1].plot(
            [row["epoch"] for row in history],
            [100 * row["val_accuracy"] for row in history],
            label=name,
        )
    axes[0].set(title="Linear-probe validation loss", xlabel="Epoch", ylabel="Cross-entropy")
    axes[1].set(
        title="Linear-probe validation accuracy", xlabel="Epoch", ylabel="Accuracy (%)"
    )
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend()
    figure.savefig(output_dir / "probe_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    summary = {
        "model_id": config.model_id,
        "train_samples": len(train_labels),
        "val_samples": len(val_labels),
        "hidden_size": train_cls.size(1),
        "cls_best_val_accuracy": max(float(row["val_accuracy"]) for row in cls_history),
        "mean_best_val_accuracy": max(float(row["val_accuracy"]) for row in mean_history),
        "config": config.as_dict(),
    }
    with (output_dir / "probe_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def load_probes(
    path: str | Path, device: torch.device
) -> tuple[nn.Linear, nn.Linear, dict[str, object]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    cls_probe = nn.Linear(int(checkpoint["hidden_size"]), int(checkpoint["num_classes"]))
    mean_probe = nn.Linear(int(checkpoint["hidden_size"]), int(checkpoint["num_classes"]))
    cls_probe.load_state_dict(checkpoint["cls_state_dict"])
    mean_probe.load_state_dict(checkpoint["mean_state_dict"])
    return cls_probe.to(device).eval(), mean_probe.to(device).eval(), checkpoint

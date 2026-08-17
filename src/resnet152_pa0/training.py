from __future__ import annotations

import copy
import time
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .utils import write_history


def _set_training_state(model: nn.Module, training: bool) -> None:
    """Keep frozen BatchNorm statistics fixed while trainable layers learn."""
    model.train(training)
    if training:
        for module in model.modules():
            parameters = list(module.parameters(recurse=False))
            if parameters and not any(p.requires_grad for p in parameters):
                module.eval()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    _set_training_state(model, training)
    loss_sum = 0.0
    correct = 0
    sample_count = 0
    start = time.perf_counter()

    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            batch_size = labels.size(0)
            loss_sum += loss.item() * batch_size
            correct += (logits.argmax(dim=1) == labels).sum().item()
            sample_count += batch_size

    return {
        "loss": loss_sum / sample_count,
        "accuracy": correct / sample_count,
        "seconds": time.perf_counter() - start,
    }


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    output_dir: str | Path,
) -> list[dict[str, float | int]]:
    output_dir = Path(output_dir)
    criterion = nn.CrossEntropyLoss()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    history: list[dict[str, float | int]] = []
    best_accuracy = -1.0
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, device)
        row: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "train_seconds": train_metrics["seconds"],
            "val_seconds": val_metrics["seconds"],
        }
        history.append(row)
        write_history(history, output_dir / "metrics.csv")
        print(
            f"epoch={epoch:02d} "
            f"train_loss={row['train_loss']:.4f} train_acc={row['train_accuracy']:.4f} "
            f"val_loss={row['val_loss']:.4f} val_acc={row['val_accuracy']:.4f}"
        )
        if val_metrics["accuracy"] > best_accuracy:
            best_accuracy = val_metrics["accuracy"]
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        torch.save(best_state, output_dir / "best_model.pt")
    return history


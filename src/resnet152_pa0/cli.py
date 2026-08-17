from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .data import build_cifar10_loaders
from .features import collect_features, plot_tsne
from .modeling import (
    build_resnet152,
    configure_trainable_layers,
    count_parameters,
    disable_skip_connections,
)
from .reporting import summarize_runs
from .training import fit
from .utils import (
    ExperimentConfig,
    environment_metadata,
    make_run_dir,
    resolve_device,
    save_json,
    seed_everything,
)


def _run_training(
    config: ExperimentConfig,
    name: str,
    pretrained: bool,
    train_mode: str,
    ablate_blocks: tuple[str, ...] = (),
) -> Path:
    seed_everything(config.seed)
    device = resolve_device(config.device)
    run_dir = make_run_dir(config, name)
    train_loader, val_loader = build_cifar10_loaders(config)
    model = build_resnet152(pretrained=pretrained)
    if ablate_blocks:
        disable_skip_connections(model, ablate_blocks)
    configure_trainable_layers(model, train_mode)
    metadata = {
        "experiment": name,
        "pretrained": pretrained,
        "train_mode": train_mode,
        "ablated_blocks": list(ablate_blocks),
        "parameters": count_parameters(model),
        "device": str(device),
        "config": config.as_dict(),
        "environment": environment_metadata(),
    }
    save_json(metadata, run_dir / "metadata.json")
    model.to(device)
    fit(
        model,
        train_loader,
        val_loader,
        device,
        config.epochs,
        config.learning_rate,
        config.weight_decay,
        run_dir,
    )
    return run_dir


def run_baseline(config: ExperimentConfig) -> None:
    _run_training(config, "baseline_head", pretrained=True, train_mode="head")


def run_ablation(config: ExperimentConfig) -> None:
    _run_training(
        config,
        "skip_ablation_head",
        pretrained=True,
        train_mode="head",
        ablate_blocks=config.ablate_blocks,
    )


def run_features(config: ExperimentConfig, checkpoint: str | None) -> None:
    seed_everything(config.seed)
    device = resolve_device(config.device)
    run_dir = make_run_dir(config, "feature_hierarchy")
    _, val_loader = build_cifar10_loaders(config)
    model = build_resnet152(pretrained=True)
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    model.to(device)
    features, labels = collect_features(model, val_loader, device, config.feature_samples)
    np.savez_compressed(run_dir / "features.npz", labels=labels, **features)
    plot_tsne(features, labels, run_dir / "tsne.png", config.seed, config.tsne_perplexity)
    save_json(
        {
            "checkpoint": checkpoint or "ImageNet weights with a new, untrained CIFAR-10 head",
            "samples": len(labels),
            "feature_shapes": {name: list(value.shape) for name, value in features.items()},
            "config": config.as_dict(),
        },
        run_dir / "metadata.json",
    )


def run_transfer(config: ExperimentConfig) -> None:
    for pretrained in (True, False):
        initialization = "pretrained" if pretrained else "random"
        for mode in ("final_block", "full"):
            _run_training(
                config,
                f"transfer_{initialization}_{mode}",
                pretrained=pretrained,
                train_mode=mode,
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PA0 Task 1 ResNet-152 experiments")
    parser.add_argument("command", choices=("baseline", "ablation", "features", "transfer", "summarize", "all"))
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML configuration")
    parser.add_argument("--checkpoint", help="Checkpoint for the feature-hierarchy experiment")
    parser.add_argument("--epochs", type=int, help="Override epoch count")
    parser.add_argument("--train-subset", type=int, help="Override training-set size")
    parser.add_argument("--val-subset", type=int, help="Override validation-set size")
    parser.add_argument("--device", help="Override device: auto, cpu, cuda, or mps")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    for field in ("epochs", "train_subset", "val_subset", "device"):
        value = getattr(args, field)
        if value is not None:
            setattr(config, field, value)

    if args.command in {"baseline", "all"}:
        run_baseline(config)
    if args.command in {"ablation", "all"}:
        run_ablation(config)
    if args.command in {"features", "all"}:
        checkpoint = args.checkpoint
        if checkpoint is None:
            candidate = Path(config.output_dir) / "baseline_head" / "best_model.pt"
            checkpoint = str(candidate) if candidate.exists() else None
        run_features(config, checkpoint)
    if args.command in {"transfer", "all"}:
        run_transfer(config)
    if args.command in {"summarize", "all"}:
        summarize_runs(config.output_dir)


if __name__ == "__main__":
    main()

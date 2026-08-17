from __future__ import annotations

import argparse
import json
from pathlib import Path

from resnet152_pa0.utils import environment_metadata, resolve_device, seed_everything

from .attention import visualize_attention
from .classification import classify_images
from .config import VitExperimentConfig
from .data import build_cifar10_loaders
from .masking import run_masking_experiment
from .modeling import load_vit, model_dimensions
from .probes import load_probes, run_probe_experiment
from .samples import download_samples


def _sample_entries(config: VitExperimentConfig) -> list[dict[str, object]]:
    return download_samples(Path(config.data_dir) / "vit_samples")


def run_classification(config: VitExperimentConfig) -> None:
    device = resolve_device(config.device)
    bundle = load_vit(config.model_id)
    results = classify_images(
        bundle, _sample_entries(config), device, Path(config.output_dir) / "classification"
    )
    for result in results:
        print(
            f"{result['image']}: {result['top1_label']} "
            f"({result['top1_probability']:.3f}), reasonable={result['appears_reasonable']}"
        )


def run_attention(config: VitExperimentConfig) -> None:
    device = resolve_device(config.device)
    bundle = load_vit(config.model_id, attentions=True)
    for entry in _sample_entries(config):
        name = Path(str(entry["path"])).stem
        visualize_attention(
            bundle, entry["path"], device, Path(config.output_dir) / "attention" / name
        )


def run_probes(config: VitExperimentConfig) -> None:
    device = resolve_device(config.device)
    bundle = load_vit(config.model_id)
    for parameter in bundle.model.parameters():
        parameter.requires_grad = False
    train_loader, val_loader = build_cifar10_loaders(config, bundle.processor)
    run_probe_experiment(
        bundle.model,
        train_loader,
        val_loader,
        config,
        device,
        Path(config.output_dir) / "probes",
    )


def run_masking(config: VitExperimentConfig) -> None:
    device = resolve_device(config.device)
    bundle = load_vit(config.model_id)
    _, val_loader = build_cifar10_loaders(config, bundle.processor)
    checkpoint_path = Path(config.output_dir) / "probes" / "linear_probes.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError("Run `vit-pa0 probes` before the masking experiment")
    cls_probe, mean_probe, checkpoint = load_probes(checkpoint_path, device)
    if checkpoint["model_id"] != config.model_id:
        raise ValueError("Probe checkpoint model does not match the configured ViT")
    run_masking_experiment(
        bundle.model,
        (cls_probe, mean_probe),
        val_loader,
        config,
        int(bundle.model.config.patch_size),
        device,
        Path(config.output_dir) / "masking",
    )


def save_run_metadata(config: VitExperimentConfig) -> None:
    bundle = load_vit(config.model_id)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "config": config.as_dict(),
                "model": model_dimensions(bundle.model),
                "environment": environment_metadata(),
            },
            handle,
            indent=2,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PA0 Task 2 Vision Transformer experiments")
    parser.add_argument("command", choices=("classify", "attention", "probes", "masking", "all"))
    parser.add_argument("--config", default="configs/vit.yaml")
    parser.add_argument("--device")
    parser.add_argument("--probe-epochs", type=int)
    parser.add_argument("--train-subset", type=int)
    parser.add_argument("--val-subset", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = VitExperimentConfig.from_yaml(args.config)
    for field in ("device", "probe_epochs", "train_subset", "val_subset"):
        value = getattr(args, field)
        if value is not None:
            setattr(config, field, value)
    seed_everything(config.seed)
    if args.command in {"classify", "all"}:
        run_classification(config)
    if args.command in {"attention", "all"}:
        run_attention(config)
    if args.command in {"probes", "all"}:
        run_probes(config)
    if args.command in {"masking", "all"}:
        run_masking(config)
    save_run_metadata(config)


if __name__ == "__main__":
    main()

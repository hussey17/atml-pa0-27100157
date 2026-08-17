from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt

from resnet152_pa0.utils import environment_metadata, resolve_device, seed_everything

from .analysis import run_analysis
from .config import VAEExperimentConfig
from .data import build_mnist_loaders
from .modeling import MNISTVAE
from .training import load_vae, train_vae


def _checkpoint_path(config: VAEExperimentConfig) -> Path:
    return Path(config.output_dir) / "training" / "best_model.pt"


def run_training(config: VAEExperimentConfig) -> None:
    seed_everything(config.seed)
    device = resolve_device(config.device)
    train_loader, validation_loader = build_mnist_loaders(config)
    model = MNISTVAE(config.input_dim, config.hidden_dim, config.latent_dim)
    train_vae(
        model,
        train_loader,
        validation_loader,
        config,
        device,
        Path(config.output_dir) / "training",
    )


def run_post_training_analysis(config: VAEExperimentConfig) -> None:
    device = resolve_device(config.device)
    checkpoint_path = _checkpoint_path(config)
    if not checkpoint_path.exists():
        raise FileNotFoundError("Run `vae-pa0 train` before `vae-pa0 analyze`")
    model, checkpoint = load_vae(checkpoint_path, device)
    _, validation_loader = build_mnist_loaders(config)
    summary = run_analysis(
        model,
        validation_loader,
        device,
        Path(config.output_dir) / "analysis",
        sample_count=config.analysis_samples,
        latent_plot_samples=config.latent_plot_samples,
        seed=config.seed,
    )
    summary["checkpoint"] = str(checkpoint_path)
    summary["best_validation_elbo"] = float(checkpoint["best_validation_elbo"])
    with (Path(config.output_dir) / "analysis" / "analysis_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)


def run_latent_sweep(config: VAEExperimentConfig) -> None:
    device = resolve_device(config.device)
    output_dir = Path(config.output_dir) / "latent_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for latent_dim in config.latent_sweep_dims:
        seed_everything(config.seed)
        sweep_config = replace(
            config, latent_dim=latent_dim, epochs=config.latent_sweep_epochs
        )
        train_loader, validation_loader = build_mnist_loaders(sweep_config)
        model = MNISTVAE(config.input_dim, config.hidden_dim, latent_dim)
        history = train_vae(
            model,
            train_loader,
            validation_loader,
            sweep_config,
            device,
            output_dir / f"dim_{latent_dim}",
            epochs=sweep_config.epochs,
        )
        best = min(history, key=lambda row: float(row["val_elbo"]))
        rows.append(
            {
                "latent_dim": latent_dim,
                "best_epoch": int(best["epoch"]),
                "best_val_elbo": float(best["val_elbo"]),
                "best_val_reconstruction": float(best["val_reconstruction"]),
                "best_val_kl": float(best["val_kl"]),
            }
        )
    with (output_dir / "latent_sweep.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    figure, axis = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    axis.plot(
        [row["latent_dim"] for row in rows],
        [row["best_val_elbo"] for row in rows],
        marker="o",
    )
    axis.set(
        title="Latent dimensionality comparison",
        xlabel="Latent dimensions",
        ylabel="Best validation negative ELBO",
    )
    axis.grid(alpha=0.2)
    figure.savefig(output_dir / "latent_sweep.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_environment(config: VAEExperimentConfig) -> None:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "environment.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {"config": config.as_dict(), "environment": environment_metadata()},
            handle,
            indent=2,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PA0 Task 4 MNIST VAE experiments")
    parser.add_argument("command", choices=("train", "analyze", "sweep", "all"))
    parser.add_argument("--config", default="configs/vae.yaml")
    parser.add_argument("--device")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--latent-dim", type=int)
    parser.add_argument("--train-subset", type=int)
    parser.add_argument("--val-subset", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = VAEExperimentConfig.from_yaml(args.config)
    for field in ("device", "epochs", "latent_dim", "train_subset", "val_subset"):
        value = getattr(args, field)
        if value is not None:
            setattr(config, field, value)
    if config.input_dim != 28 * 28:
        raise ValueError("MNIST experiments require input_dim=784")
    seed_everything(config.seed)
    if args.command in {"train", "all"}:
        run_training(config)
    if args.command in {"analyze", "all"}:
        run_post_training_analysis(config)
    if args.command == "sweep":
        run_latent_sweep(config)
    save_environment(config)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from resnet152_pa0.utils import environment_metadata, resolve_device, seed_everything

from .alignment import (
    fit_orthogonal_alignment,
    save_alignment_metrics,
    save_alignment_visualization,
)
from .config import ClipExperimentConfig
from .data import build_stl10_loader
from .embeddings import ImageEmbeddings, extract_image_embeddings
from .evaluation import classification_metrics, save_zero_shot_results
from .modeling import ClipBundle, load_openai_clip
from .modality import analyze_modality_gap
from .prompts import PROMPT_TEMPLATES, STL10_CLASSES, encode_text_prototypes


def extract_test_embeddings(
    config: ClipExperimentConfig,
    bundle: ClipBundle,
    device: torch.device,
) -> ImageEmbeddings:
    loader = build_stl10_loader(
        config,
        bundle.preprocess,
        split="test",
        subset_count=config.test_subset,
        seed_offset=1,
    )
    embeddings = extract_image_embeddings(bundle.model, loader, device)
    embeddings.save(Path(config.output_dir) / "embeddings" / "stl10_test.npz")
    return embeddings


def run_zero_shot(
    config: ClipExperimentConfig,
    bundle: ClipBundle,
    device: torch.device,
    test_embeddings: ImageEmbeddings | None = None,
) -> ImageEmbeddings:
    test_embeddings = test_embeddings or extract_test_embeddings(config, bundle, device)
    results = {}
    for strategy in PROMPT_TEMPLATES:
        _, prototypes, prompt_bank = encode_text_prototypes(
            bundle.model, bundle.tokenize, STL10_CLASSES, strategy, device
        )
        metrics = classification_metrics(
            test_embeddings.normalized, test_embeddings.labels, prototypes, STL10_CLASSES
        )
        metrics["prompts"] = prompt_bank
        results[strategy] = metrics
        print(
            f"{strategy:>11}: {metrics['accuracy']:.4f} "
            f"({metrics['correct']}/{metrics['samples']})"
        )
    save_zero_shot_results(results, Path(config.output_dir) / "zero_shot")
    return test_embeddings


def run_modality_gap(
    config: ClipExperimentConfig,
    bundle: ClipBundle,
    device: torch.device,
) -> None:
    loader = build_stl10_loader(
        config,
        bundle.preprocess,
        split="test",
        balanced_count=config.gap_samples,
        seed_offset=2,
    )
    images = extract_image_embeddings(bundle.model, loader, device)
    raw_text, _, prompt_bank = encode_text_prototypes(
        bundle.model,
        bundle.tokenize,
        STL10_CLASSES,
        config.gap_prompt_strategy,
        device,
    )
    paired_text = raw_text[images.labels]
    output_dir = Path(config.output_dir) / "modality_gap"
    metrics = analyze_modality_gap(
        images.raw.numpy(),
        paired_text.numpy(),
        images.labels.numpy(),
        output_dir,
        seed=config.seed,
        perplexity=config.tsne_perplexity,
    )
    metrics["prompt_strategy"] = config.gap_prompt_strategy
    metrics["prompts"] = prompt_bank
    with (output_dir / "modality_gap_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(
        "normalized paired cosine similarity: "
        f"{metrics['l2_normalized']['mean_paired_cosine_similarity']:.4f}"
    )


def run_alignment(
    config: ClipExperimentConfig,
    bundle: ClipBundle,
    device: torch.device,
    test_embeddings: ImageEmbeddings | None = None,
) -> ImageEmbeddings:
    fit_loader = build_stl10_loader(
        config,
        bundle.preprocess,
        split="train",
        balanced_count=config.alignment_samples,
        seed_offset=3,
    )
    fit_images = extract_image_embeddings(bundle.model, fit_loader, device)
    _, text_prototypes, prompt_bank = encode_text_prototypes(
        bundle.model,
        bundle.tokenize,
        STL10_CLASSES,
        config.alignment_prompt_strategy,
        device,
    )
    paired_fit_text = text_prototypes[fit_images.labels].numpy()
    rotation, diagnostics = fit_orthogonal_alignment(
        fit_images.normalized.numpy(), paired_fit_text
    )
    test_embeddings = test_embeddings or extract_test_embeddings(config, bundle, device)
    aligned_test = torch.from_numpy(test_embeddings.normalized.numpy() @ rotation).float()
    before = classification_metrics(
        test_embeddings.normalized, test_embeddings.labels, text_prototypes, STL10_CLASSES
    )
    after = classification_metrics(
        aligned_test, test_embeddings.labels, text_prototypes, STL10_CLASSES
    )
    output_dir = Path(config.output_dir) / "alignment"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "rotation.npy", rotation)
    save_alignment_visualization(
        fit_images.normalized.numpy(),
        fit_images.normalized.numpy() @ rotation,
        paired_fit_text,
        fit_images.labels.numpy(),
        output_dir,
        seed=config.seed,
        perplexity=config.tsne_perplexity,
    )
    metrics = {
        "fit_split": "train",
        "evaluation_split": "test" if config.test_subset is None else "test subset",
        "fit_samples": int(len(fit_images.labels)),
        "evaluation_samples": int(len(test_embeddings.labels)),
        "prompt_strategy": config.alignment_prompt_strategy,
        "prompts": prompt_bank,
        "diagnostics": diagnostics,
        "classification_before": before,
        "classification_after": after,
        "accuracy_change_percentage_points": 100.0 * (after["accuracy"] - before["accuracy"]),
    }
    save_alignment_metrics(metrics, output_dir)
    print(
        f"alignment accuracy: {before['accuracy']:.4f} -> {after['accuracy']:.4f} "
        f"({metrics['accuracy_change_percentage_points']:+.2f} pp)"
    )
    return test_embeddings


def save_environment(config: ClipExperimentConfig, device: torch.device) -> None:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "environment.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "config": config.as_dict(),
                "resolved_device": str(device),
                "environment": environment_metadata(),
                "implementation": "OpenAI CLIP (https://github.com/openai/CLIP)",
            },
            handle,
            indent=2,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PA0 Task 3 OpenAI CLIP experiments")
    parser.add_argument("command", choices=("zero-shot", "gap", "align", "all"))
    parser.add_argument("--config", default="configs/clip.yaml")
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--test-subset", type=int)
    parser.add_argument("--gap-samples", type=int)
    parser.add_argument("--alignment-samples", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = ClipExperimentConfig.from_yaml(args.config)
    for field in ("device", "batch_size", "test_subset", "gap_samples", "alignment_samples"):
        value = getattr(args, field)
        if value is not None:
            setattr(config, field, value)
    config.validate()
    seed_everything(config.seed)
    device = resolve_device(config.device)
    bundle = load_openai_clip(config.model_name, device, config.download_root)
    test_embeddings = None
    if args.command in {"zero-shot", "all"}:
        test_embeddings = run_zero_shot(config, bundle, device, test_embeddings)
    if args.command in {"gap", "all"}:
        run_modality_gap(config, bundle, device)
    if args.command in {"align", "all"}:
        run_alignment(config, bundle, device, test_embeddings)
    save_environment(config, device)


if __name__ == "__main__":
    main()

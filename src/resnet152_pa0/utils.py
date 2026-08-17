from __future__ import annotations

import csv
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


@dataclass
class ExperimentConfig:
    seed: int = 42
    data_dir: str = "data"
    output_dir: str = "outputs"
    device: str = "auto"
    num_workers: int = 2
    batch_size: int = 32
    image_size: int = 224
    epochs: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    train_subset: int | None = None
    val_subset: int | None = None
    feature_samples: int = 2000
    tsne_perplexity: float = 30.0
    ablate_blocks: tuple[str, ...] = ("layer2.1", "layer3.1", "layer4.1")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            values = yaml.safe_load(handle) or {}
        if "ablate_blocks" in values:
            values["ablate_blocks"] = tuple(values["ablate_blocks"])
        return cls(**values)

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["ablate_blocks"] = list(self.ablate_blocks)
        return result


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Determinism is preferred for a coursework comparison. Some CUDA kernels
    # can still differ across PyTorch/CUDA versions, so versions are recorded.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_run_dir(config: ExperimentConfig, name: str) -> Path:
    path = Path(config.output_dir) / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(value: Any, path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)


def write_history(rows: list[dict[str, Any]], path: str | Path) -> None:
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def environment_metadata() -> dict[str, Any]:
    return {
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
    }


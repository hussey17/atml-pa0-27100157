from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class VAEExperimentConfig:
    seed: int = 42
    data_dir: str = "data"
    output_dir: str = "outputs/vae"
    device: str = "auto"
    num_workers: int = 2
    batch_size: int = 128
    epochs: int = 15
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    input_dim: int = 784
    hidden_dim: int = 400
    latent_dim: int = 2
    beta: float = 1.0
    train_subset: int | None = None
    val_subset: int | None = None
    analysis_samples: int = 10
    latent_plot_samples: int = 5000
    latent_sweep_dims: tuple[int, ...] = (2, 10, 30)
    latent_sweep_epochs: int = 5

    @classmethod
    def from_yaml(cls, path: str | Path) -> "VAEExperimentConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            values = yaml.safe_load(handle) or {}
        if "latent_sweep_dims" in values:
            values["latent_sweep_dims"] = tuple(int(value) for value in values["latent_sweep_dims"])
        return cls(**values)

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["latent_sweep_dims"] = list(self.latent_sweep_dims)
        return result

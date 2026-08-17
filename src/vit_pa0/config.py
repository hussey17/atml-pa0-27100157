from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class VitExperimentConfig:
    seed: int = 42
    model_id: str = "WinKawaks/vit-tiny-patch16-224"
    data_dir: str = "data"
    output_dir: str = "outputs/vit"
    device: str = "auto"
    num_workers: int = 2
    batch_size: int = 64
    probe_epochs: int = 20
    probe_learning_rate: float = 1e-3
    probe_weight_decay: float = 1e-4
    train_subset: int | None = None
    val_subset: int | None = None
    mask_fractions: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 0.75)
    random_mask_trials: int = 3

    @classmethod
    def from_yaml(cls, path: str | Path) -> "VitExperimentConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            values = yaml.safe_load(handle) or {}
        if "mask_fractions" in values:
            values["mask_fractions"] = tuple(float(x) for x in values["mask_fractions"])
        return cls(**values)

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["mask_fractions"] = list(self.mask_fractions)
        return result

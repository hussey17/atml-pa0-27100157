from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ClipExperimentConfig:
    seed: int = 42
    model_name: str = "ViT-B/32"
    data_dir: str = "data"
    output_dir: str = "outputs/clip"
    download_root: str | None = None
    device: str = "auto"
    num_workers: int = 2
    batch_size: int = 64
    test_subset: int | None = None
    gap_samples: int = 100
    alignment_samples: int = 100
    gap_prompt_strategy: str = "photo"
    alignment_prompt_strategy: str = "photo"
    tsne_perplexity: float = 25.0

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ClipExperimentConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls(**(yaml.safe_load(handle) or {}))

    def validate(self) -> None:
        if not 50 <= self.gap_samples <= 100:
            raise ValueError("gap_samples must be between 50 and 100, as required by Task 3")
        if self.alignment_samples < 10:
            raise ValueError("alignment_samples must be at least 10")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.test_subset is not None and self.test_subset < 1:
            raise ValueError("test_subset must be positive or null")
        if self.tsne_perplexity <= 0:
            raise ValueError("tsne_perplexity must be positive")
        strategies = {"plain", "photo", "descriptive"}
        if self.gap_prompt_strategy not in strategies:
            raise ValueError(f"Unknown gap_prompt_strategy: {self.gap_prompt_strategy}")
        if self.alignment_prompt_strategy not in strategies:
            raise ValueError(
                f"Unknown alignment_prompt_strategy: {self.alignment_prompt_strategy}"
            )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

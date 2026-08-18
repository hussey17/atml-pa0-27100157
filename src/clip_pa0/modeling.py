from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch


@dataclass
class ClipBundle:
    model: torch.nn.Module
    preprocess: Callable
    tokenize: Callable
    model_name: str


def load_openai_clip(
    model_name: str,
    device: torch.device,
    download_root: str | Path | None = None,
) -> ClipBundle:
    """Load a model through OpenAI's official ``clip`` package."""
    try:
        import clip
    except ImportError as error:
        raise ImportError(
            "OpenAI CLIP is not installed. Run `python -m pip install -e .`."
        ) from error
    if not all(hasattr(clip, name) for name in ("load", "tokenize", "available_models")):
        raise ImportError(
            "The imported `clip` module is not OpenAI CLIP. Remove the unrelated PyPI "
            "package and install https://github.com/openai/CLIP.git."
        )
    root = str(download_root) if download_root else None
    model, preprocess = clip.load(
        model_name,
        device=device,
        jit=False,
        download_root=root,
    )
    model.eval()
    return ClipBundle(
        model=model,
        preprocess=preprocess,
        tokenize=clip.tokenize,
        model_name=model_name,
    )

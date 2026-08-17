from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from transformers import AutoImageProcessor, AutoModelForImageClassification


@dataclass
class VitBundle:
    model: nn.Module
    processor: object


def load_vit(model_id: str, *, attentions: bool = False) -> VitBundle:
    """Load one ImageNet ViT and its exact paired image processor."""
    processor = AutoImageProcessor.from_pretrained(model_id)
    kwargs = {"attn_implementation": "eager"} if attentions else {}
    model = AutoModelForImageClassification.from_pretrained(model_id, **kwargs)
    model.eval()
    return VitBundle(model=model, processor=processor)


def get_backbone(model: nn.Module) -> nn.Module:
    backbone = getattr(model, "base_model", None)
    if backbone is None or backbone is model:
        backbone = getattr(model, "vit", None)
    if backbone is None:
        raise AttributeError("Could not locate the ViT backbone on the classification model")
    return backbone


def final_tokens(model: nn.Module, pixel_values: torch.Tensor) -> torch.Tensor:
    """Return `[batch, 1 + patches, hidden]` from the final encoder layer."""
    outputs = get_backbone(model)(pixel_values=pixel_values, return_dict=True)
    return outputs.last_hidden_state


def pool_tokens(tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if tokens.ndim != 3 or tokens.size(1) < 2:
        raise ValueError("Expected [batch, tokens, hidden] with a CLS and at least one patch")
    return tokens[:, 0], tokens[:, 1:].mean(dim=1)


def cls_patch_attention(attentions: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    """Return mean-head and per-head final-layer CLS-to-patch attention."""
    if not attentions:
        raise ValueError("No attentions returned; load the model with eager attention")
    final = attentions[-1]
    if final.ndim != 4 or final.size(-1) != final.size(-2):
        raise ValueError(f"Unexpected attention shape: {tuple(final.shape)}")
    per_head = final[:, :, 0, 1:]
    patch_count = per_head.size(-1)
    grid = int(patch_count**0.5)
    if grid * grid != patch_count:
        raise ValueError(f"Patch count {patch_count} is not a square grid")
    return per_head.mean(dim=1).reshape(-1, grid, grid), per_head.reshape(final.size(0), final.size(1), grid, grid)


def model_dimensions(model: nn.Module) -> dict[str, int]:
    config = model.config
    return {
        "image_size": int(config.image_size),
        "patch_size": int(config.patch_size),
        "hidden_size": int(config.hidden_size),
        "layers": int(config.num_hidden_layers),
        "heads": int(config.num_attention_heads),
        "labels": int(config.num_labels),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }

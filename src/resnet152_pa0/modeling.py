from __future__ import annotations

import types
from collections.abc import Iterable

import torch
from torch import nn
from torchvision.models import ResNet152_Weights, resnet152
from torchvision.models.resnet import Bottleneck, ResNet


def build_resnet152(num_classes: int = 10, pretrained: bool = True) -> ResNet:
    """Build ResNet-152 and replace its 1000-way ImageNet classifier."""
    weights = ResNet152_Weights.DEFAULT if pretrained else None
    model = resnet152(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def configure_trainable_layers(model: ResNet, mode: str) -> list[nn.Parameter]:
    """Select parameters for head-only, final-block, or full fine-tuning."""
    if mode not in {"head", "final_block", "full"}:
        raise ValueError(f"Unknown fine-tuning mode: {mode}")

    for parameter in model.parameters():
        parameter.requires_grad = mode == "full"

    if mode in {"head", "final_block"}:
        for parameter in model.fc.parameters():
            parameter.requires_grad = True
    if mode == "final_block":
        for parameter in model.layer4.parameters():
            parameter.requires_grad = True

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters were selected")
    return trainable


def _block_without_skip(self: Bottleneck, x: torch.Tensor) -> torch.Tensor:
    """Torchvision Bottleneck forward pass with the identity addition removed."""
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)
    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)
    out = self.conv3(out)
    out = self.bn3(out)
    # Deliberately omit: out += identity
    return self.relu(out)


def get_module(model: nn.Module, path: str) -> nn.Module:
    module: nn.Module = model
    for part in path.split("."):
        if part.isdigit():
            module = module[int(part)]  # type: ignore[index]
        else:
            module = getattr(module, part)
    return module


def disable_skip_connections(model: ResNet, block_paths: Iterable[str]) -> None:
    """Disable identity additions in shape-preserving Bottleneck blocks.

    Transition blocks (index 0 in each stage) usually contain downsampling and
    cannot safely drop the identity in a controlled ablation, so they are rejected.
    The architecture and state-dict keys remain unchanged.
    """
    for path in block_paths:
        block = get_module(model, path)
        if not isinstance(block, Bottleneck):
            raise TypeError(f"{path!r} is not a torchvision Bottleneck")
        if block.downsample is not None:
            raise ValueError(f"{path!r} is a transition block; choose a shape-preserving block")
        block.forward = types.MethodType(_block_without_skip, block)


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image

from .modeling import VitBundle, cls_patch_attention


def _processed_image(pixel_values: torch.Tensor, processor: object) -> np.ndarray:
    mean = torch.tensor(processor.image_mean, dtype=pixel_values.dtype).view(3, 1, 1)
    std = torch.tensor(processor.image_std, dtype=pixel_values.dtype).view(3, 1, 1)
    image = pixel_values.detach().cpu() * std + mean
    return image.clamp(0, 1).permute(1, 2, 0).numpy()


def _normalize_map(values: np.ndarray) -> np.ndarray:
    minimum, maximum = float(values.min()), float(values.max())
    return (values - minimum) / (maximum - minimum + 1e-12)


def _head_statistics(per_head: torch.Tensor) -> dict[str, object]:
    flat = per_head.flatten(1)
    probabilities = flat / flat.sum(dim=1, keepdim=True).clamp_min(1e-12)
    entropies = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1) / math.log(flat.size(1))
    normalized = functional.normalize(flat, dim=1)
    similarities = normalized @ normalized.T
    heads = flat.size(0)
    mean_similarity = float((similarities.sum() - heads) / max(heads * (heads - 1), 1))
    maxima = [divmod(int(index), per_head.size(-1)) for index in flat.argmax(dim=1)]
    return {
        "normalized_entropy_per_head": entropies.tolist(),
        "mean_pairwise_cosine_similarity": mean_similarity,
        "peak_patch_row_col_per_head": [list(position) for position in maxima],
    }


def visualize_attention(
    bundle: VitBundle,
    image_path: str | Path,
    device: torch.device,
    output_dir: str | Path,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image = Image.open(image_path).convert("RGB")
    pixel_values = bundle.processor(images=image, return_tensors="pt")["pixel_values"].to(device)
    bundle.model.to(device).eval()
    with torch.inference_mode():
        outputs = bundle.model(pixel_values=pixel_values, output_attentions=True, return_dict=True)
    mean_map, per_head = cls_patch_attention(outputs.attentions)
    display_image = _processed_image(pixel_values[0], bundle.processor)
    height, width = display_image.shape[:2]
    upsampled = functional.interpolate(
        mean_map[:, None], size=(height, width), mode="bilinear", align_corners=False
    )[0, 0].cpu().numpy()
    normalized_map = _normalize_map(upsampled)

    figure, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    axes[0].imshow(display_image)
    axes[0].set_title("Model input")
    axes[1].imshow(mean_map[0].cpu(), cmap="inferno")
    axes[1].set_title("Final-layer CLS attention")
    axes[2].imshow(display_image)
    axes[2].imshow(normalized_map, cmap="Reds", alpha=0.55)
    axes[2].set_title("Attention overlay")
    for axis in axes:
        axis.axis("off")
    figure.savefig(output_dir / "attention_overlay.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    head_count = per_head.size(1)
    columns = min(4, head_count)
    rows = math.ceil(head_count / columns)
    figure, axes = plt.subplots(
        rows, columns, figsize=(3.5 * columns, 3.5 * rows), squeeze=False, constrained_layout=True
    )
    for index, axis in enumerate(axes.flat):
        axis.axis("off")
        if index < head_count:
            axis.imshow(per_head[0, index].cpu(), cmap="inferno")
            axis.set_title(f"Head {index}")
    figure.suptitle("Final-layer CLS-to-patch attention by head")
    figure.savefig(output_dir / "attention_heads.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    prediction = int(outputs.logits.argmax(dim=-1)[0])
    stats = {
        "image": str(image_path),
        "prediction": bundle.model.config.id2label[prediction],
        "attention_tensor_shape": list(outputs.attentions[-1].shape),
        "patch_grid": list(mean_map.shape[-2:]),
        **_head_statistics(per_head[0].cpu()),
    }
    with (output_dir / "attention_stats.json").open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)
    return stats

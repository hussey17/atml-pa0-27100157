from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image

from .modeling import VitBundle, model_dimensions


def classify_images(
    bundle: VitBundle,
    image_entries: Iterable[dict[str, object]],
    device: torch.device,
    output_dir: str | Path,
) -> list[dict[str, object]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle.model.to(device).eval()
    results = []
    for entry in image_entries:
        path = Path(str(entry["path"]))
        image = Image.open(path).convert("RGB")
        pixel_values = bundle.processor(images=image, return_tensors="pt")["pixel_values"].to(device)
        with torch.inference_mode():
            probabilities = bundle.model(pixel_values=pixel_values).logits.softmax(dim=-1)[0]
        values, indices = probabilities.topk(5)
        labels = [bundle.model.config.id2label[int(index)] for index in indices]
        keywords = [str(x).lower() for x in entry.get("reasonable_keywords", [])]
        top1 = labels[0]
        results.append(
            {
                "image": str(path),
                "expected_object": entry.get("expected_object"),
                "top1_label": top1,
                "top1_probability": float(values[0]),
                "appears_reasonable": any(keyword in top1.lower() for keyword in keywords) if keywords else None,
                "top5": [
                    {"label": label, "probability": float(probability)}
                    for label, probability in zip(labels, values)
                ],
            }
        )
    payload = {
        "model": bundle.model.config.name_or_path,
        "dimensions": model_dimensions(bundle.model),
        "predictions": results,
    }
    with (output_dir / "predictions.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return results

from __future__ import annotations

import json
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class SampleSpec:
    filename: str
    url: str
    expected_object: str
    reasonable_keywords: tuple[str, ...]


SAMPLES = (
    SampleSpec(
        filename="coco_cats.jpg",
        url="http://images.cocodataset.org/val2017/000000039769.jpg",
        expected_object="two domestic cats",
        reasonable_keywords=("cat", "tabby", "lynx", "tiger"),
    ),
    SampleSpec(
        filename="parrots.png",
        url="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/parrots.png",
        expected_object="parrots",
        reasonable_keywords=("parrot", "macaw", "cockatoo", "lorikeet"),
    ),
)


def download_samples(root: str | Path) -> list[dict[str, object]]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    manifest = []
    for sample in SAMPLES:
        path = root / sample.filename
        if not path.exists():
            request = urllib.request.Request(sample.url, headers={"User-Agent": "pa0-vit-coursework/0.1"})
            with urllib.request.urlopen(request, timeout=60) as response, path.open("wb") as handle:
                handle.write(response.read())
        entry = asdict(sample)
        entry["reasonable_keywords"] = list(sample.reasonable_keywords)
        entry["path"] = str(path)
        manifest.append(entry)
    with (root / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch


STL10_CLASSES = (
    "airplane",
    "bird",
    "car",
    "cat",
    "deer",
    "dog",
    "horse",
    "monkey",
    "ship",
    "truck",
)

PROMPT_TEMPLATES: dict[str, tuple[str, ...]] = {
    "plain": ("{label}",),
    "photo": ("a photo of {article} {label}.",),
    "descriptive": (
        "a clear photo of {article} {label}.",
        "a close-up photo of {article} {label}.",
        "a centered photo of {article} {label}.",
        "a high-quality photo of {article} {label}.",
        "a natural image containing {article} {label}.",
    ),
}


def _article(label: str) -> str:
    return "an" if label[0].lower() in "aeiou" else "a"


def prompts_for_class(label: str, strategy: str) -> list[str]:
    if strategy not in PROMPT_TEMPLATES:
        raise KeyError(
            f"Unknown prompt strategy {strategy!r}; choose from {tuple(PROMPT_TEMPLATES)}"
        )
    return [
        template.format(label=label, article=_article(label))
        for template in PROMPT_TEMPLATES[strategy]
    ]


def l2_normalize(values: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return values / values.norm(dim=-1, keepdim=True).clamp_min(eps)


@torch.inference_mode()
def encode_text_prototypes(
    model: torch.nn.Module,
    tokenize: Callable,
    class_names: Sequence[str],
    strategy: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, list[str]]]:
    """Return raw and normalized class prototypes plus the exact prompt bank.

    Each individual prompt is normalized before template ensembling, following
    CLIP's cosine-similarity geometry. The raw prototype is retained solely for
    the required pre-normalization modality-gap comparison.
    """
    raw_prototypes = []
    normalized_prototypes = []
    prompt_bank: dict[str, list[str]] = {}
    for label in class_names:
        prompts = prompts_for_class(label, strategy)
        prompt_bank[label] = prompts
        tokens = tokenize(prompts).to(device)
        features = model.encode_text(tokens).float()
        raw_prototypes.append(features.mean(dim=0))
        normalized_prototypes.append(l2_normalize(features).mean(dim=0))
    raw = torch.stack(raw_prototypes).cpu()
    normalized = l2_normalize(torch.stack(normalized_prototypes)).cpu()
    return raw, normalized, prompt_bank

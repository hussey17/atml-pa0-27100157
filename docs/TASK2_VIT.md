# Task 2 - Understanding Vision Transformers

This document maps every required Task 2 experiment to the implementation and explains the model mechanics needed to interpret the eventual results. The code uses [`WinKawaks/vit-tiny-patch16-224`](https://huggingface.co/WinKawaks/vit-tiny-patch16-224), a 5.7M-parameter ImageNet ViT converted from the timm checkpoint. It is small enough for local experiments while preserving the standard ViT layout: one CLS token and a 14x14 grid of patch tokens.

## Requirement-to-code map

| Manual requirement | Code | Main output |
|---|---|---|
| Classify 1-3 images with an ImageNet ViT | `classification.py`, `samples.py` | `classification/predictions.json` |
| Final-layer CLS-to-patch attention overlay | `attention.py`, `cls_patch_attention()` | overlay and attention-head PNGs |
| Analyze head specialization | `_head_statistics()` | entropy, peak locations, pairwise similarity |
| Random and centered patch masking | `masking.py` | accuracy CSV and robustness plot |
| CLS versus mean-pooled linear probes | `probes.py` | metrics, plot, and probe checkpoint |

## How this ViT works

The processor resizes/crops each RGB image to 224x224 and applies the normalization paired with the checkpoint. A learned convolution with kernel and stride 16 divides the image into non-overlapping patches:

```text
224 / 16 = 14 patches per side
14 x 14 = 196 image patches
196 patch tokens + 1 learned CLS token = 197 tokens
```

Each flattened patch is projected into the model's hidden dimension. A learned positional embedding is added because self-attention alone does not encode patch order. The token sequence then passes through transformer encoder blocks containing LayerNorm, multi-head self-attention, an MLP, and residual connections.

For one attention head, the input token matrix is projected to queries, keys, and values:

```text
Q = X W_Q,  K = X W_K,  V = X W_V
Attention(Q,K,V) = softmax(Q K^T / sqrt(d_head)) V
```

The dot products score how strongly each query token relates to every key token. Scaling by the square root of head width prevents softmax saturation. Different heads have different projection matrices and can therefore attend to different spatial or semantic patterns. The final classifier reads the last-layer CLS representation.

## 1. ImageNet classification

`vit-pa0 classify` downloads two stable demonstration images (cats and parrots), processes them with the checkpoint's own `AutoImageProcessor`, and records top-1 plus top-5 predictions. The JSON also records the expected visible object and a conservative keyword-based `appears_reasonable` flag. This flag is a convenience, not a substitute for inspecting the image and discussing the result in the report.

## 2-3. Patch attention and interpretation

The attention command loads the model with eager attention so Hugging Face returns post-softmax attention tensors. For the final encoder block the tensor is:

```text
[batch, heads, query tokens, key tokens]
```

`cls_patch_attention()` selects query index 0 (CLS), removes key index 0 (CLS itself), averages across heads, and reshapes 196 values to 14x14. Bilinear upsampling maps this grid to 224x224 before a red heat map is overlaid on the exact processed image.

The output includes all individual heads. Head specialization is assessed rather than assumed:

- normalized entropy measures whether a head is diffuse or concentrated;
- peak patch coordinates show whether heads select different regions;
- mean pairwise cosine similarity measures how alike their spatial maps are.

Lower cross-head similarity, distinct peaks, and meaningfully different map shapes are evidence of specialization. Attention is not guaranteed to be a faithful causal explanation: a high weight indicates routing inside one layer, while Grad-CAM uses output gradients to measure class-sensitive feature-map influence. ViT attention is available without backpropagation and has direct token-to-token structure, but it should still be treated as diagnostic evidence rather than proof of what caused a prediction.

## 4. Patch masking

The masking experiment evaluates classification accuracy on CIFAR-10 using the frozen linear probes from experiment 5. This makes accuracy well-defined without requiring the restricted ImageNet validation set. Selected 16x16 regions are replaced with zero in normalized pixel space, corresponding to the processor's mean color.

- Random masking samples patch locations independently with a recorded seed and averages multiple trials.
- Center masking chooses patches nearest the 14x14 grid center deterministically.

Both CLS and mean-pooled probes are evaluated at mask fractions 0%, 10%, 25%, 50%, and 75%. The clean 0% point anchors the curves. A center mask can be more damaging when CIFAR objects are centered; random masks spread information loss and may leave enough object evidence. At high fractions, both should degrade. These are hypotheses to compare against the generated CSV, not pre-written findings.

The experiment sets pixels to the mean instead of deleting sequence positions. This preserves positional indices and tensor shapes, so the comparison isolates missing visual content rather than changing the transformer architecture.

## 5. CLS versus mean-pooled linear probes

The ViT backbone is frozen. For each CIFAR-10 image, the final hidden states produce two representations:

```text
CLS feature  = final_tokens[:, 0]
Mean feature = mean(final_tokens[:, 1:], dimension=patches)
```

Independent `Linear(hidden_size, 10)` classifiers are trained with the same optimizer, seed, train/validation split, and number of epochs. Because only a linear layer learns, the comparison tests how linearly accessible class information is under each pooling rule.

CLS pooling often benefits when the pretraining classifier explicitly optimized the CLS token. Mean pooling can be competitive when pretraining distributed information across patch tokens, such as reconstruction- or patch-level objectives. The observed winner must come from `probe_metrics.csv`.

## Setup and commands

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .

# ImageNet predictions for two sample images
vit-pa0 classify --config configs/vit.yaml

# Averaged and per-head attention visualizations
vit-pa0 attention --config configs/vit.yaml

# Extract frozen CIFAR-10 features and train both probes
vit-pa0 probes --config configs/vit.yaml

# Requires the probe checkpoint from the previous command
vit-pa0 masking --config configs/vit.yaml
```

`vit-pa0 all` runs the complete workflow in dependency order. A quick pipeline check is:

```bash
vit-pa0 probes --probe-epochs 1 --train-subset 64 --val-subset 32 --device cpu
vit-pa0 masking --val-subset 32 --device cpu
```

The first run downloads the ViT checkpoint, CIFAR-10, and two sample images. `device: auto` chooses CUDA, then Apple MPS, then CPU. Full frozen-feature extraction is much faster on a GPU.

## Outputs for the write-up

```text
outputs/vit/
  metadata.json
  classification/predictions.json
  attention/<image>/attention_overlay.png
  attention/<image>/attention_heads.png
  attention/<image>/attention_stats.json
  probes/probe_metrics.csv
  probes/probe_summary.json
  probes/probe_comparison.png
  probes/linear_probes.pt
  masking/masking_metrics.csv
  masking/masking_comparison.png
```

For the later analysis, use the actual outputs to discuss prediction plausibility, foreground/background attention, head diversity, robustness slope under each mask strategy, and the probe accuracy gap. Record hardware, subset sizes, trial count, and any deviations from `configs/vit.yaml`.

## Code layout

```text
src/vit_pa0/modeling.py        checkpoint loading, tokens, pooling, attention extraction
src/vit_pa0/classification.py  top-k ImageNet inference
src/vit_pa0/attention.py       overlays, head grids, specialization statistics
src/vit_pa0/probes.py          feature extraction and frozen linear probes
src/vit_pa0/masking.py         random/center masking and robustness evaluation
src/vit_pa0/data.py            deterministic CIFAR-10 processing/loaders
src/vit_pa0/cli.py             experiment orchestration
```

## References

- Dosovitskiy et al., [An Image Is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)
- [Hugging Face ViT documentation](https://huggingface.co/docs/transformers/model_doc/vit)
- [ViT tiny checkpoint model card](https://huggingface.co/WinKawaks/vit-tiny-patch16-224)
- Jain and Wallace, [Attention is not Explanation](https://arxiv.org/abs/1902.10186)

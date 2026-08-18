# Task 3 - CLIP on STL-10

This guide maps every Task 3 requirement to executable code and explains the model, prompt comparison, modality-gap measurements, and orthogonal alignment. The default configuration uses OpenAI's official `ViT-B/32` CLIP checkpoint and evaluates all 8,000 STL-10 test images. Generated metrics belong in the later results write-up; the explanations here distinguish mathematical expectations from observations.

## Requirement-to-code map

| Manual requirement | Implementation | Main output |
|---|---|---|
| Download STL-10 through Torchvision | `data.py::build_stl10_loader()` | `data/stl10_binary/` |
| Load official OpenAI CLIP | `modeling.py::load_openai_clip()` | cached `ViT-B/32` checkpoint |
| Compare three prompt strategies on the complete test set | `prompts.py`, `cli.py::run_zero_shot()` | accuracy CSV/JSON and bar plot |
| Extract 50-100 paired image/text embeddings | balanced sampler plus `extract_image_embeddings()` | compressed embeddings |
| Compare raw and normalized modality distributions with t-SNE | `modality.py::analyze_modality_gap()` | metrics JSON, NPZ, and two-panel plot |
| Learn an orthogonal Procrustes map | `alignment.py::fit_orthogonal_alignment()` | rotation matrix and fit diagnostics |
| Visualize alignment and recompute test accuracy | `cli.py::run_alignment()` | shared t-SNE plot and before/after metrics |

## How CLIP works

CLIP is a dual encoder. Its vision transformer maps an image to a vector, while its text transformer maps a token sequence to a vector in the same dimensional space:

```text
STL-10 image -> official CLIP preprocessing -> ViT-B/32 -> image vector v
text prompt  -> byte-pair tokenizer          -> Transformer -> text vector t

v_hat = v / ||v||_2
t_hat = t / ||t||_2
similarity(image, text) = v_hat dot t_hat
```

During pretraining, a batch of paired images and captions produces an image-by-text similarity matrix. Symmetric contrastive cross-entropy raises the score of each true pair and lowers scores for mismatches, once with images as queries and once with texts as queries. This does not force the two encoders to produce identical distributions. It trains their relative directions to make paired concepts more similar than competing concepts.

At zero-shot inference, the ten class prompts act as classifier weights. For image vector `v_hat` and class prototype `t_hat_c`, the predicted class is:

```text
prediction = argmax_c (v_hat dot t_hat_c)
```

The learned logit scale is irrelevant to `argmax`, so the implementation uses the normalized cosine scores directly. No STL-10 classifier is trained in Part 1.

## 1. Zero-shot classification and prompting

`PROMPT_TEMPLATES` defines the three required strategies:

- `plain`: the class alone, such as `cat`.
- `photo`: a natural-language prompt, such as `a photo of a cat.`
- `descriptive`: an ensemble of five variants covering clear, close-up, centered, high-quality, and natural images.

For the descriptive strategy, every prompt is encoded and L2-normalized. The prompt vectors for a class are averaged and normalized again. This creates one class prototype without unfairly giving a class more logits merely because it has more templates. The exact generated prompt bank is saved in `metrics.json`, including correct `a`/`an` articles.

The default `test_subset: null` is deliberate: it evaluates the complete STL-10 test split. The image encoder runs once, and all three strategies reuse those embeddings. `classification_metrics()` records overall accuracy, per-class accuracy, and a fixed 10-by-10 confusion matrix. The later report should compare actual measurements instead of assuming that longer prompts always win; prompt ensembling can help, have little effect, or introduce a mismatch.

## 2. Exploring the modality gap

The code chooses 100 test images reproducibly and approximately equally across all ten classes. Each image embedding is paired with the text embedding for its ground-truth label under the configured `photo` strategy. It records the raw encoder outputs and their row-wise L2-normalized versions.

For both versions, `modality_gap_statistics()` computes:

- mean image and text vector norms;
- Euclidean distance between the modality centroids;
- mean Euclidean distance between paired embeddings;
- mean paired cosine similarity.

Two independent joint t-SNE fits visualize raw image/text points and normalized image/text points. Circles are images, crosses are paired label texts, and color denotes class. t-SNE emphasizes local neighborhoods and distorts global geometry, so visible island separation is not a quantitative distance. The JSON measures must be discussed alongside the plot.

L2 normalization removes encoder-specific magnitude, which can substantially change Euclidean gaps. It does not change the cosine angle of an individual pair. CLIP can classify well even when image and text clouds have different centroids because classification depends on the ranking of image-to-class cosine similarities, not on equality of the marginal image and text distributions.

## 3. Orthogonal Procrustes alignment

The alignment solves the assignment's constrained problem:

```text
R* = argmin_R ||X R - Y||_F    subject to R^T R = I
```

`scipy.linalg.orthogonal_procrustes()` obtains the closed-form rotation/reflection from the singular value decomposition of `X^T Y`. The returned scale diagnostic is not applied: Task 3 requests an orthogonal transform, and multiplying only by `R` preserves vector lengths and image-to-image inner products.

To avoid fitting on evaluation examples, `run_alignment()` builds `X` from a balanced 100-image subset of the STL-10 **training** split and pairs it with ground-truth class text prototypes in `Y`. It then applies the learned `R` to the separate complete test split. This makes the before/after accuracy comparison more credible than learning and evaluating the rotation on the same images.

The output records:

- `||R^T R - I||_F` to verify orthogonality;
- paired cosine and Frobenius residual before/after on the alignment-fit examples;
- original and aligned complete-test accuracy, per-class accuracy, and confusion matrices;
- the change in percentage points.

One t-SNE model is fit jointly to original image, aligned image, and paired text embeddings. The two panels then share coordinates, avoiding the false impression that axes from unrelated t-SNE fits are comparable. Procrustes is guaranteed to minimize its training-pair Frobenius objective, but it is **not** guaranteed to improve held-out classification. A decrease would be a valid result: class-label prototypes provide a low-rank, repeated target and may rotate away useful relationships learned from CLIP's much broader caption data.

## Setup and commands

Python 3.10+ is required. A CUDA GPU is recommended for the full 8,000-image evaluation, though CPU and Apple MPS are supported by the device resolver.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

The editable install fetches `clip` from OpenAI's official GitHub repository. The first experiment downloads the official model weights and STL-10. Run stages separately or together:

```bash
clip-pa0 zero-shot --config configs/clip.yaml
clip-pa0 gap      --config configs/clip.yaml
clip-pa0 align    --config configs/clip.yaml
clip-pa0 all      --config configs/clip.yaml
```

For a pipeline check, use a small test subset and fewer alignment examples while retaining the manual's minimum 50 modality samples:

```bash
clip-pa0 all --device cpu --batch-size 8 --test-subset 64 \
  --gap-samples 50 --alignment-samples 20
```

Do not use subset results as the required final zero-shot comparison. Restore `test_subset: null` for the report run. `configs/clip.yaml` contains every default, including the seed, model, prompts used for pairing, sample counts, and t-SNE perplexity.

## Outputs and write-up checklist

```text
outputs/clip/
├── environment.json
├── embeddings/stl10_test.npz
├── zero_shot/
│   ├── accuracy.csv
│   ├── metrics.json
│   └── prompt_accuracy.png
├── modality_gap/
│   ├── modality_gap_metrics.json
│   ├── modality_gap_embeddings.npz
│   └── modality_gap_tsne.png
└── alignment/
    ├── rotation.npy
    ├── alignment_metrics.json
    ├── alignment_tsne_coordinates.npz
    └── alignment_tsne.png
```

For the later write-up, report the evaluation sample count to prove that all 8,000 test images were used. Compare the three exact accuracies, interpret class-specific changes, quantify raw versus normalized gaps, and compare the Procrustes residual and held-out accuracy before/after. State the model, seed, prompt strategy used to learn `R`, fitting split, evaluation split, and hardware. Do not turn the conceptual expectations in this guide into observed claims without the generated metrics.

## Reproducibility and limitations

The sampler, Torch, NumPy, and t-SNE use the configured seed. Exact floating-point values can vary by hardware and library version, so `environment.json` records the resolved device and package versions. The images for t-SNE and alignment are class-balanced to avoid a visualization dominated by sampling imbalance.

Important limitations are the small alignment set, repeated class-label text targets, t-SNE's local/distorted view, prompt sensitivity, and the mismatch between natural-caption pretraining and ten short STL-10 labels. The Procrustes rotation is a diagnostic post-hoc map, not a replacement for CLIP's learned joint geometry.

## References

- [OpenAI CLIP implementation](https://github.com/openai/CLIP)
- Radford et al., [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- [Torchvision STL-10 documentation](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.STL10.html)
- [SciPy orthogonal Procrustes documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.orthogonal_procrustes.html)
- [scikit-learn t-SNE documentation](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)

# PA0 Experiments - ResNet-152, ViT, CLIP, and VAE

Reproducible PyTorch experiments for Tasks 1-4 of EE-5102/CS-6304 Assignment 0. The project records machine-readable metrics and metadata so the later NeurIPS-format analysis can be based on actual runs rather than copied console output.

- **Task 1 - ResNet-152:** the implementation and model guide continue below.
- **Task 2 - Vision Transformer:** see the complete [ViT implementation guide](docs/TASK2_VIT.md), covering ImageNet classification, attention overlays, head specialization, patch masking, and CLS-versus-mean linear probes.
- **Task 3 - CLIP:** see the complete [CLIP implementation guide](docs/TASK3_CLIP.md), covering three zero-shot prompt strategies on STL-10, raw/normalized modality-gap analysis, and held-out Orthogonal Procrustes alignment.
- **Task 4 - Variational Autoencoder:** see the complete [MNIST VAE implementation guide](docs/TASK4_VAE.md), covering ELBO training, latent visualization, reconstruction, generation, dimensionality sweeps, and the Doersch comparison.

## Task 3 quick start

Task 3 uses OpenAI's official `ViT-B/32` CLIP package and Torchvision STL-10. The default configuration evaluates all 8,000 test images, selects 100 balanced examples for the modality plot, and learns the Procrustes rotation from a separate balanced subset of the training split.

```bash
python -m pip install -e .
clip-pa0 zero-shot --config configs/clip.yaml
clip-pa0 gap --config configs/clip.yaml
clip-pa0 align --config configs/clip.yaml
```

`clip-pa0 all` runs the complete task and reuses the full-test image embeddings between the zero-shot and alignment stages. See [docs/TASK3_CLIP.md](docs/TASK3_CLIP.md) for the model mechanics, requirement-to-code map, output schema, interpretation guidance, and a reduced pipeline-check command.

## Task 1 - Inner Workings of ResNet-152

> Scope: required Task 1 items 1-4. The optional t-SNE/UMAP comparison, confusion analysis, and ResNet-18 comparison are intentionally not included yet.

## What is implemented

| Manual requirement | Implementation | Output |
|---|---|---|
| Pretrained ResNet-152, CIFAR-10 head, frozen backbone | `build_resnet152()` and `configure_trainable_layers(..., "head")` | `outputs/baseline_head/` |
| Remove selected residual connections and retrain head | `disable_skip_connections()` | `outputs/skip_ablation_head/` |
| Capture early, middle, and late representations | Forward hooks on `layer1`, `layer3`, and `avgpool` | `features.npz`, `tsne.png` |
| Pretrained vs random initialization | `run_transfer()` executes both initializations | Four transfer run folders |
| Final block vs full-backbone fine-tuning | `final_block` and `full` trainable modes | Metrics and best checkpoint per run |
| Compare training dynamics and validation accuracy | `summarize_runs()` | `comparison.csv`, `training_curves.png` |

## How ResNet-152 works

ResNet-152 is a convolutional network with a stem followed by four residual stages. Torchvision uses the bottleneck layout `[3, 8, 36, 3]`: 3 blocks in `layer1`, 8 in `layer2`, 36 in `layer3`, and 3 in `layer4`. Each bottleneck applies `1x1 -> 3x3 -> 1x1` convolutions. The first and last `1x1` layers reduce and then restore channel width, making the expensive `3x3` convolution narrower.

For an input `x`, an ordinary residual block computes:

```text
F(x) = BN3(Conv1x1(ReLU(BN2(Conv3x3(ReLU(BN1(Conv1x1(x))))))))
y    = ReLU(F(x) + identity(x))
```

The identity is either `x` itself or a learned projection in a transition block. The additive path lets a gradient reach earlier layers through a short derivative term in addition to the long residual branch. This makes identity mappings easy to represent and reduces optimization degradation in very deep networks.

The model pipeline in this repository is:

```text
CIFAR-10 image (32x32)
  -> resize/augment to 224x224 and ImageNet-normalize
  -> 7x7 convolution, BatchNorm, ReLU, max pool
  -> layer1: 3 bottlenecks, 256 channels       [early hook]
  -> layer2: 8 bottlenecks, 512 channels
  -> layer3: 36 bottlenecks, 1024 channels     [middle hook]
  -> layer4: 3 bottlenecks, 2048 channels
  -> global average pool                       [late hook]
  -> Linear(2048, 10)
  -> CIFAR-10 logits
```

### Preprocessing and transfer learning

CIFAR-10 images are resized because the ImageNet-pretrained model learned at a much larger spatial scale. The inputs use the ImageNet channel mean and standard deviation expected by the weights. Training data receives random resized crops and horizontal flips; validation data is deterministic. See `cifar10_transforms()` in `src/resnet152_pa0/data.py`.

The original 1000-class fully connected layer is replaced by `Linear(2048, 10)` in `build_resnet152()`. In the baseline, all backbone parameters have `requires_grad=False`; only the new head is optimized. Frozen BatchNorm modules are kept in evaluation mode so their pretrained running statistics are not silently modified.

Training a 60-million-parameter ResNet-152 from scratch on only 50,000 CIFAR-10 training images is usually unnecessary and impractical: it is compute-heavy, has much greater capacity than the dataset supports, and discards useful edge, texture, shape, and object-part features learned from ImageNet. If a small trained head performs well, that is direct evidence that the frozen feature space is transferable. It does not prove every feature is optimal for CIFAR-10; later fine-tuning experiments measure how much task-specific adaptation helps.

### Controlled residual ablation

`disable_skip_connections()` replaces the forward behavior only for named, shape-preserving bottlenecks. The selected defaults are `layer2.1`, `layer3.1`, and `layer4.1`. Transition blocks such as `layer2.0` are rejected because they change spatial size/channels and use a projection; removing such a path would combine architectural shape changes with the residual-path question.

The ablated block computes `ReLU(F(x))` instead of `ReLU(F(x) + x)`. It retains the same weights and state-dict keys, isolating the effect of the addition as closely as possible. Because the manual asks to retrain only the modified head here, backbone gradients are not measured in this experiment. The gradient-flow explanation is architectural: without the additive identity route, information and gradients must pass through every nonlinear residual branch, typically slowing optimization and weakening the pretrained representation.

### Feature hierarchy

Forward hooks collect activations without modifying the model. Four-dimensional feature maps are global-average-pooled before t-SNE so each image contributes one vector:

- `layer1` (early): local edges, colors, and simple textures.
- `layer3` (middle): motifs, parts, and more contextual combinations.
- `avgpool` (late): compact semantic representation used by the classifier.

t-SNE is fit independently at each depth and plotted with class labels. Interpret local neighborhoods and cluster overlap; do not compare global distances or apparent cluster sizes literally across separate t-SNE fits.

### Fine-tuning matrix

The `transfer` command runs the required 2x2 comparison:

| Initialization | Trainable region | Question answered |
|---|---|---|
| ImageNet | `layer4` + head | Strong transfer with moderate compute |
| ImageNet | Full backbone | Maximum pretrained adaptation |
| Random | `layer4` + head | Whether a random frozen lower backbone is useful |
| Random | Full backbone | From-scratch reference |

The expected best compute/accuracy trade-off is often pretrained final-block fine-tuning, but the write-up should state what the recorded results show. Full fine-tuning can improve accuracy at much higher memory/time cost. The early layers are generally most reusable because basic edge/texture detectors are less dataset-specific; later layers encode ImageNet categories more strongly.

## Setup

Python 3.10+ is required. A GPU is strongly recommended for the full transfer matrix.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

The first run downloads CIFAR-10 and, for pretrained experiments, the official Torchvision weights.

## Run the experiments

Run each part separately so failures or long runs are easy to resume:

```bash
# 1. Frozen-backbone baseline
resnet152-pa0 baseline --config configs/default.yaml

# 2. Identical setup with selected identity additions removed
resnet152-pa0 ablation --config configs/default.yaml

# 3. Features from the best baseline checkpoint, if it exists
resnet152-pa0 features --config configs/default.yaml

# 4. Four transfer-learning configurations
resnet152-pa0 transfer --config configs/default.yaml

# Summarize whichever training runs have completed
resnet152-pa0 summarize --config configs/default.yaml
```

`resnet152-pa0 all` executes the complete sequence. For a pipeline check before committing expensive compute:

```bash
resnet152-pa0 baseline --epochs 1 --train-subset 64 --val-subset 32 --device cpu
```

Edit `configs/default.yaml` for batch size, workers, sample counts, output path, and ablated block names. On Apple Silicon, `device: auto` selects MPS; on a CUDA machine it selects CUDA.

## Outputs and later analysis

Every training directory contains:

- `metrics.csv`: epoch-level loss, accuracy, and elapsed time for train/validation.
- `metadata.json`: initialization, trainable mode, parameter counts, device, versions, seed, and full configuration.
- `best_model.pt`: state dict at the highest observed validation accuracy.

The feature run contains `features.npz`, `tsne.png`, and metadata. The `summarize` command writes a cross-run `comparison.csv` plus train/validation learning curves to `outputs/`. Generated data, weights, and outputs are intentionally ignored by Git because they are large and machine-specific.

For the later report, compare the same columns across runs and discuss:

1. baseline convergence and validation accuracy;
2. baseline versus skip-ablation convergence, final accuracy, and runtime;
3. how visible class overlap changes from early to late t-SNE plots;
4. accuracy/compute trade-offs across the four transfer settings;
5. limitations: CIFAR-to-ImageNet resize mismatch, few epochs, t-SNE stochasticity, and hardware/version effects.

Do not claim the expected explanations above as observed results until the experiments have been run.

## Reproducibility and tests

The seed controls Python, NumPy, PyTorch, dataset subsetting, and training shuffling. The project records library/device metadata, uses deterministic cuDNN settings, and preserves a fixed validation transform. Exact floating-point equality across hardware is not guaranteed.

```bash
python -m pytest -q
```

## Repository layout

```text
configs/default.yaml              experiment defaults
configs/vit.yaml                  ViT experiment defaults
configs/clip.yaml                 CLIP/STL-10 experiment defaults
configs/vae.yaml                  VAE experiment defaults
src/resnet152_pa0/data.py         CIFAR-10 transforms and loaders
src/resnet152_pa0/modeling.py     model, freezing, residual ablation
src/resnet152_pa0/training.py     train/evaluate loops and checkpoints
src/resnet152_pa0/features.py     hooks and t-SNE visualization
src/resnet152_pa0/reporting.py    cross-run tables and learning curves
src/resnet152_pa0/cli.py          experiment orchestration
src/vit_pa0/                      Task 2 ViT experiments
src/clip_pa0/                     Task 3 CLIP experiments
src/vae_pa0/                      Task 4 VAE experiments
docs/TASK2_VIT.md                 Task 2 model/code guide
docs/TASK3_CLIP.md                Task 3 model/code guide
docs/TASK4_VAE.md                 Task 4 model/code guide
tests/test_modeling.py            architecture and ablation tests
tests/test_vit_core.py            ViT unit tests
tests/test_clip_core.py           CLIP/alignment unit tests
tests/test_vae.py                 VAE unit tests
```

## References

- He et al., [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- [Torchvision ResNet-152 documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet152.html)
- [PyTorch transfer learning tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [PyTorch forward-hook documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook)
- [scikit-learn t-SNE documentation](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)

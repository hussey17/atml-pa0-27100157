# Task 4 - Variational Autoencoders on MNIST

This guide maps every required VAE task to executable PyTorch code and explains the probabilistic model behind it. The default configuration trains a 2D-latent MLP VAE for 15 epochs, which directly supports the assignment's latent-space plot.

## Requirement-to-code map

| Manual requirement | Implementation | Output |
|---|---|---|
| Encoder returns mean and log-variance | `MNISTVAE.encode()` | `mu`, `logvar` tensors |
| Reparameterization trick | `MNISTVAE.reparameterize()` | differentiable latent sample `z` |
| Bernoulli decoder | decoder logits plus sigmoid probabilities | 784 pixel probabilities |
| BCE reconstruction plus Gaussian KL | `negative_elbo()` | ELBO and separate components |
| Adam training for 10-20 epochs | `train_vae()`; default 15 epochs | metrics, curve, best checkpoint |
| Plot posterior means by digit | `plot_latent_space()` | PNG and NPZ embeddings |
| Show 5-10 reconstructions | `reconstruction_analysis()` | paired figure and per-class errors |
| Draw new samples from the prior | `generation_analysis()` | generated grid and sampled latent vectors |
| Compare latent dimensionalities | `vae-pa0 sweep` | CSV and comparison plot |
| Compare with Doersch | comparison section below | report-ready technical basis |

## Model and probabilistic assumptions

The observed image is `x in [0,1]^784`. The encoder is an amortized approximation to the otherwise intractable posterior:

```text
x (784) -> Linear(784, 400) -> ReLU
        -> mu(x)      in R^d
        -> logvar(x)  in R^d

q_phi(z | x) = Normal(mu(x), diag(exp(logvar(x))))
```

The standard deviation is `exp(0.5 * logvar)`. Sampling is rewritten as a deterministic differentiable transformation of parameter-free noise:

```text
epsilon ~ Normal(0, I)
z = mu + exp(0.5 * logvar) * epsilon
```

This is the reparameterization trick. Randomness is isolated in `epsilon`, so gradients can flow through `mu` and `logvar` into the encoder.

The decoder maps a latent sample to one logit per pixel:

```text
z (d) -> Linear(d, 400) -> ReLU -> Linear(400, 784) -> logits
p_theta(x_j = 1 | z) = sigmoid(logit_j)
```

Conditioned on `z`, pixels are modeled as independent Bernoulli variables. The implementation retains logits and uses `binary_cross_entropy_with_logits`, which combines sigmoid and cross-entropy in a numerically stable operation.

MNIST pixels are grayscale values rather than literal binary observations. Following the assignment and Doersch tutorial, the BCE can be interpreted as the expected log-likelihood of a randomly binarized image. This is a standard educational VAE setup, though richer image likelihoods are possible.

## Negative ELBO

The objective minimized per image is:

```text
negative ELBO = reconstruction BCE + beta * KL(q_phi(z|x) || Normal(0,I))

KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
```

With the default `beta=1`, this is the assignment's ordinary VAE objective. The reconstruction term encourages faithful decoded pixels. The KL term regularizes each approximate posterior toward the standard-normal prior, making prior samples usable by the decoder and encouraging a continuous latent space. Too little KL can produce disconnected encodings; too much can make the decoder ignore `z` (posterior collapse).

`negative_elbo()` sums over pixels and latent coordinates for each image, then averages across the batch. Therefore the recorded units are nats per image and do not silently change with batch size.

## Training behavior and checkpoints

`train_vae()` uses Adam at `1e-3`, minibatches of 128, and 15 epochs by default. It records train and validation values for:

- total negative ELBO;
- Bernoulli reconstruction term;
- KL divergence.

The checkpoint with the lowest validation negative ELBO is saved. Separating the components matters: a falling total objective can hide KL collapse, while an extremely large KL may indicate poor prior matching or unstable optimization.

## Post-training analysis

### Latent space

The encoder's posterior mean, not a noisy sample, represents every MNIST test image. For `d=2`, those means are plotted directly and colored by the true digit. For a higher-dimensional run, the code applies PCA to two dimensions and records that reduction in metadata. The raw means, reduced coordinates, and labels are retained in `latent_embeddings.npz`.

Meaningful structure can appear as same-digit neighborhoods and smooth transitions, but overlap is expected because the VAE is trained without class labels. A 2D bottleneck prioritizes visual continuity over perfectly separated classes.

### Reconstructions

Ten deterministic reconstructions decode `mu(x)` rather than a sampled `z`, preventing sampling noise from obscuring reconstruction quality. The analysis also measures pixel BCE and MSE for every test digit separately. Those per-class metrics provide evidence for identifying digits reconstructed better or worse, while the figure reveals blur, missing strokes, and ambiguous shapes.

### New generations

Ten independent vectors are sampled from `Normal(0,I)` and decoded. The exact latent vectors are saved beside the generated grid for reproducibility. Valid but ambiguous in-between digits are plausible because a VAE learns a continuous density and BCE often averages multiple likely stroke configurations.

## Comparison with Carl Doersch's implementation

The comparison below is based on the tutorial's public `mnist_vae.prototxt`, not an assumed generic VAE.

| Aspect | This repository | Doersch tutorial implementation |
|---|---|---|
| Framework | PyTorch | Caffe with custom VAE layers |
| Encoder | `784 -> 400 -> (mu, logvar)` | `784 -> 1000 -> 500 -> 250 ->` Gaussian parameters |
| Decoder | `d -> 400 -> 784` | `30 -> 250 -> 500 -> 1000 -> 784` |
| Activations | ReLU hidden layers | ReLU hidden layers |
| Default latent size | 2, chosen for direct visualization | 30 |
| Observation model | Bernoulli pixel probabilities | Bernoulli interpretation via sigmoid cross-entropy |
| Optimizer | Adam | Adam |

The implementations agree on the diagonal-Gaussian posterior, standard-normal prior, reparameterized sampling, Bernoulli/sigmoid-cross-entropy output, ReLU MLPs, and Adam. This repository is deliberately shallower to match the assignment's suggested 400-unit baseline and to run quickly on CPU. Doersch's network is considerably deeper and wider.

Doersch reports that performance is fairly insensitive to latent dimensionality over a broad middle range, with problems at extremes. The default 2D choice here sacrifices capacity for interpretability. `vae-pa0 sweep` trains matched models at dimensions 2, 10, and 30 so the eventual report can compare reconstruction, KL, and validation ELBO instead of repeating the tutorial's claim without evidence.

For the results comparison, use the generated figures. Look for the tutorial's noted failure mode: realistic samples mixed with ambiguous digits that resemble transitions between classes. Do not claim similarity or superiority until the full run has completed.

## Setup and commands

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .

# Train the default 2D VAE for 15 epochs
vae-pa0 train --config configs/vae.yaml

# Create latent, reconstruction, and generation analyses
vae-pa0 analyze --config configs/vae.yaml

# Train and analyze in dependency order
vae-pa0 all --config configs/vae.yaml

# Optional matched latent-size comparison
vae-pa0 sweep --config configs/vae.yaml
```

A fast pipeline check is:

```bash
vae-pa0 all --epochs 1 --train-subset 1024 --val-subset 256 --device cpu
```

The first run downloads MNIST. `device: auto` chooses CUDA, then Apple MPS, then CPU. Edit `configs/vae.yaml` to change latent size, hidden width, beta, sample counts, or sweep settings.

## Outputs for the later write-up

```text
outputs/vae/
  environment.json
  training/
    metrics.csv
    learning_curves.png
    best_model.pt
    metadata.json
  analysis/
    latent_space.png
    latent_embeddings.npz
    reconstructions.png
    reconstruction_metrics_by_class.json
    generated_samples.png
    generation_metadata.json
    analysis_summary.json
  latent_sweep/
    latent_sweep.csv
    latent_sweep.png
    dim_<d>/...
```

For the final report, use actual values and figures to answer: Did negative ELBO decrease? Did the KL remain nonzero? Which digits overlap in latent space? Which classes have the largest reconstruction error? Are generated samples diverse and recognizable? How does latent size affect reconstruction and prior regularization?

## Code layout

```text
src/vae_pa0/modeling.py   encoder, posterior, reparameterization, decoder, ELBO
src/vae_pa0/data.py       deterministic MNIST loaders and subsets
src/vae_pa0/training.py   optimization, validation, metrics, plots, checkpoints
src/vae_pa0/analysis.py   latent plot, reconstructions, per-class errors, generations
src/vae_pa0/cli.py        train, analyze, sweep, and all workflows
tests/test_vae.py         shapes, reparameterization, KL, and beta tests
```

## References

- Kingma and Welling, [An Introduction to Variational Autoencoders](https://arxiv.org/abs/1906.02691)
- Doersch, [Tutorial on Variational Autoencoders](https://arxiv.org/abs/1606.05908)
- Doersch, [reference Caffe implementation](https://github.com/cdoersch/vae_tutorial)
- PyTorch, [binary cross-entropy with logits](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.binary_cross_entropy_with_logits.html)

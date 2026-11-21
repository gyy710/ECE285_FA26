# ECE285_FA26 — HW1: Chest X-Ray VAE

A PyTorch convolutional variational autoencoder (VAE) for unconditional chest X-ray generation, with four experiments comparing image resolution (64 or 128 pixels) and KL weight (`beta=1` or `0.1`). NORMAL and PNEUMONIA images are pooled for training; class labels are used only for splitting, statistics, and fixed comparison examples.

**Data and trained weights are not bundled.** Download the dataset and train the models before running generation, comparisons, or evaluation. `data/`, `outputs/`, virtual environments, and large checkpoint files are excluded from Git.

## Setup

Run commands from the repository root with Python 3.12:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-metrics.txt
```

`requirements.txt` contains training and plotting dependencies; `requirements-metrics.txt` adds Ignite 0.5.5 and SciPy. To install the exact versions recorded for these experiments instead, use:

```bash
python -m pip install -r requirements-lock.txt
```

The recorded environment used PyTorch 2.14.1, Torchvision 0.29.1, Ignite 0.5.5, and SciPy 1.18.1 on an Apple M4 Max with MPS. The lock file records that environment; availability and numerical results can differ across platforms.

```bash
python -c 'import torch, torchvision; print("torch:", torch.__version__); print("torchvision:", torchvision.__version__); print("MPS available:", torch.backends.mps.is_available())'
```

Training configs default to `mps`. On a machine without MPS, append `--device cpu` to training and model-check commands. Generation and comparison scripts also accept `--device cpu`; evaluation uses `--feature-device cpu`. This implementation supports CPU and MPS.

## Dataset and split

Download the [Chest X-Ray Images (Pneumonia) dataset on Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) and extract it into this layout:

```text
data/chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

`data.root` in each config must point directly to the directory containing `train/`, `val/`, and `test/`. The loader selects those top-level splits, avoids nested duplicate dataset copies, and filters hidden metadata files such as `._*.jpeg`. The original dataset contains 5,216 training, 16 validation, and 624 test images.

Because the original validation set is small, `data.validation_fraction=0.1` holds out approximately 10% of each training class's **candidate filename groups**, using seed 42. Files stay in their original locations. The original 16 validation images are unused; the original test split is retained.

| Experimental split | NORMAL | PNEUMONIA | Total |
| --- | ---: | ---: | ---: |
| Train | 1,207 | 3,492 | 4,699 |
| Validation | 134 | 383 | 517 |
| Test | 234 | 390 | 624 |

NORMAL groups preserve the full `IM-####` or `NORMAL2-IM-####` prefix. PNEUMONIA images sharing `personN` are grouped together across bacteria/virus filenames. These are unverified identity proxies: numbers may be reused. Train and validation share no candidate groups, but this does not establish patient independence. Ambiguous filename overlaps with the unchanged test split are recorded in the manifest and do not, by themselves, establish patient duplication.

Preprocessing converts images to one grayscale channel, resizes to the model's native square resolution, and produces `float32` tensors in `[0,1]`, without random augmentation. To inspect data and save a preview and split manifest under `outputs/baseline/`:

```bash
python dataset.py
```

The baseline batch shape is `(64, 1, 64, 64)`. Every run saves `split_manifest.json` with relative paths, class counts, filename groups, unused original validation files, and a split fingerprint. Validation paths still begin with `train/` because they were held out from the original training directory. Setting `validation_fraction=0` restores the original split; use a new experiment directory if changing the split.

## Model and objective

The encoder has four `4×4`, stride-2, padding-1 convolutions with channels `32, 64, 128, 256` and ReLU activations. Two linear heads produce 128-dimensional `mu` and `logvar`. The decoder mirrors the encoder with a linear projection and transposed convolutions; its final output is logits. There is no BatchNorm or Dropout.

```text
x → encoder → mu, logvar
z = mu + exp(0.5 * logvar) * epsilon,  epsilon ~ N(0, I)
z → decoder → logits → sigmoid → image
```

The 64-pixel model has a `256×4×4` bottleneck and 2,955,201 parameters. The 128-pixel model has a `256×8×8` bottleneck and 7,686,081 parameters. Thus the resolution comparison also changes the linear layers' capacity.

```text
loss = a * R + beta * KL
R    = sum_pixels BCEWithLogits(logits, x)
KL   = -0.5 * sum_latent(1 + logvar - mu² - exp(logvar))
```

Both terms are summed within each image and then averaged over the batch. Grayscale `[0,1]` values are soft BCE targets. Loss uses decoder logits directly; sigmoid is applied for image generation and display.

For 64-pixel runs, `a=1`. For 128-pixel runs, `a=0.25`, so `a × image_size² = 4096` for both resolutions. The logged reconstruction column is already scaled by `a`. The 128-pixel, `beta=0.1` objective is `0.25 R_128 + 0.1 KL`; this is a deliberately rescaled objective and should not be described as an unchanged image-likelihood ELBO. Weighted total losses across different beta values are not directly comparable, and native BCE curves have different image targets across resolutions.

For stable reconstruction comparisons, scripts decode the posterior mean: `decode(mu).sigmoid()`. The model's ordinary `forward()` samples even in evaluation mode. Prior generation instead decodes fresh `z ~ N(0,I)` and requires no input image.

## Train the four experiments

All runs use seed 42, latent dimension 128, batch size 64, Adam with learning rate `0.001`, the same split, and a 30-epoch budget.

| Run | Config | Resolution | beta | a | Output directory | Recorded best epoch |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| Baseline | `config.json` | 64 | 1 | 1 | `outputs/baseline/` | 27 |
| Lower beta | `configs/beta_0p1.json` | 64 | 0.1 | 1 | `outputs/beta_0p1/` | 29 |
| Higher resolution | `configs/resolution_128.json` | 128 | 0.1 | 0.25 | `outputs/resolution_128/` | 30 |
| Higher resolution, beta 1 | `configs/resolution_128_beta_1.json` | 128 | 1 | 0.25 | `outputs/resolution_128_beta_1/` | 29 |

From a fresh clone with the dataset installed:

```bash
python train.py --config config.json --epochs 30 --output-dir outputs/baseline
python train.py --config configs/beta_0p1.json --epochs 30 --output-dir outputs/beta_0p1
python train.py --config configs/resolution_128.json --epochs 30 --output-dir outputs/resolution_128
python train.py --config configs/resolution_128_beta_1.json --epochs 30 --output-dir outputs/resolution_128_beta_1
```

Each run writes `best.pt` (lowest validation total), `last.pt`, `history.csv`, loss curves, fixed-seed prior and reconstruction previews, configuration/environment snapshots, and the split manifest. Checkpoints include optimizer, history, and RNG state. Validation uses fixed posterior noise from a separate CPU generator, and epoch averages weight batches by their actual image counts; final short batches are retained.

As an alternative to the first training command, start with a smoke run and continue it:

```bash
python train.py --epochs 1 --output-dir outputs/smoke
python train.py --epochs 30 --resume outputs/smoke/last.pt --output-dir outputs/baseline
```

`--epochs` is the total target, including restored epochs. Existing training artifacts require `--resume` or a new output directory; incompatible settings or split fingerprints are rejected. The recorded baseline and 128-pixel/beta-0.1 run each continued from a one-epoch smoke checkpoint. Cross-device or cross-version runs are not guaranteed to be bitwise identical.

## Generate and compare

After training, the default generation command loads `outputs/baseline/best.pt`, exports 64 prior samples with seed 123, and optionally saves a separate validation reconstruction grid:

```bash
python generate.py --reconstruct
python generate.py --checkpoint outputs/resolution_128/best.pt --reconstruct
```

The default destination is `generation/` beside the checkpoint. It contains individual grayscale PNGs, `generated_grid.png`, optional `reconstruction_grid.png`, and `metadata.json` with seeds, checkpoint hash, and reconstruction sources. Pure prior generation does not need the dataset; `--reconstruct` verifies the recorded validation split. A separate CPU generator draws all latent vectors before batching, so changing the batch size preserves those draws.

To use another seed or sample count, choose a new destination:

```bash
python generate.py --checkpoint outputs/baseline/best.pt --num-samples 128 --seed 124 --output-dir outputs/baseline/generation_seed124
```

The comparisons use matched **epoch-30 `last.pt` checkpoints**, the same validation files, and seed-123 latent draws:

```bash
# Beta comparison at 64 pixels.
python compare_experiments.py --device cpu

# Resolution comparison at beta=0.1; computes all-validation common-target MAE/MSE.
python compare_resolutions.py --device cpu

# Beta comparison at 128 pixels.
python compare_experiments.py --baseline outputs/resolution_128_beta_1/last.pt --experiment outputs/resolution_128/last.pt --output-dir outputs/comparison_beta_128 --device cpu

# Resolution comparison at beta=1.
python compare_resolutions.py --baseline outputs/baseline/last.pt --experiment outputs/resolution_128_beta_1/last.pt --output-dir outputs/comparison_resolution_beta_1 --device cpu
```

The first two destinations default to `outputs/comparison_beta/` and `outputs/comparison_resolution/`. Each comparison writes reconstruction and prior-generation grids, validation curves, and a provenance summary. Fixed reconstruction previews use the first two filenames per class. Common-target errors use all 517 validation images: each model receives its native input, 64-pixel predictions are bilinearly upsampled (`align_corners=False`), and both predictions are compared with the same real 128-pixel target. The final five-image batch is included, for 8,470,528 target pixels.

Generation, comparison, and evaluation commands refuse nonempty output destinations. Use a new `--output-dir` when repeating them.

## Recorded results

All four rows below use epoch 30. Validation reconstruction and KL are the fixed-noise Monte Carlo values recorded during training; common-target MAE/MSE use deterministic posterior-mean reconstructions.

| Resolution | beta | a | Validation scaled BCE | Validation KL | Common-target MAE | Common-target MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 1 | 1 | 2370.2084 | 22.3691 | 0.0610582 | 0.00633054 |
| 64 | 0.1 | 1 | 2348.0380 | 86.2804 | 0.0428444 | 0.00355928 |
| 128 | 1 | 0.25 | 2363.6442 | 24.0997 | 0.0605499 | 0.00618809 |
| 128 | 0.1 | 0.25 | 2339.2657 | 94.9506 | 0.0386648 | 0.00302152 |

At beta 0.1, the higher-resolution model reduced common-target MAE by about 9.76% and MSE by 15.11%. At beta 1, the MSE reduction was about 2.25%. These are validation reconstruction results; they do not establish an equivalent improvement in prior-generated images. Samples retain overall chest structure but remain smooth and lack reliable fine rib texture.

There is **one training seed**, with no uncertainty intervals. The fourth run (128 pixels, beta 1) was added after the initial 128-pixel/beta-0.1 test evaluation. Its epoch-30 checkpoint was retained for the descriptive comparison; test scores were not used to select a checkpoint epoch or adjust parameters. The experiment sequence therefore should not be presented as a fully preregistered four-model test comparison.

## FID and Inception Score

Only the two 128-pixel models were evaluated for FID / IS. The original evaluation used the validation-selected `outputs/resolution_128/best.pt`, which is also epoch 30. The additional beta-1 evaluation used `last.pt` at epoch 30, rather than its best-validation epoch 29.

```bash
# Default checkpoint: outputs/resolution_128/best.pt.
python evaluate.py --feature-device mps
python evaluate.py --checkpoint outputs/resolution_128_beta_1/last.pt --feature-device mps
```

Use `--feature-device cpu` without MPS. Evaluation may download official pretrained Inception-v3 weights on first use. Each command defaults to `evaluation/` beside its checkpoint; `--seed`, `--batch-size`, and `--output-dir` are available overrides. Changing the recorded protocol should be disclosed with new results.

The recorded protocol uses **all 624 test images and 624 fresh prior samples, seed 456**, with batch size 32 and the final 16-image batch retained. It does not reuse the 64 seed-123 display samples or evaluate reconstructions.

| 128-pixel model, epoch 30 | FID: Ignite 1000 probabilities | Global Inception Score |
| --- | ---: | ---: |
| beta 1, a 0.25 | 0.2086336724 | 1.6416618928 |
| beta 0.1, a 0.25 | 0.1364407317 | 1.8399837316 |

These values follow the assignment's [Ignite evaluation tutorial](https://pytorch-ignite.ai/blog/gan-evaluation-with-fid-and-is/) implementation with explicitly documented preprocessing. **Ignite 0.5.5's default extractor returns 1000 post-softmax ImageNet probabilities. This FID is not conventional 2048-dimensional pool3 FID and cannot be compared numerically with it.**

Real and generated images follow the same feature-input pipeline:

```text
native-resolution grayscale float [0,1]
→ ToPILImage: multiply by 255 and truncate to uint8
→ direct PIL BILINEAR resize to 299×299, without cropping
→ RGB with three identical channels → ToTensor
→ ImageNet Normalize(mean=.485/.456/.406, std=.229/.224/.225)
→ pretrained Inception-v3 with transform_input=True
```

Explicit ImageNet normalization corrects the tutorial's omitted input standardization; this is not a literal reproduction of its preprocessing. The script does not use the weights' resize/center-crop transform, per-image contrast adjustment, or min-max rescaling.

One shared Inception network computes probabilities once per image. Cached CPU `float32` probabilities are passed to Ignite FID and InceptionScore through an identity feature extractor without applying another softmax; Ignite accumulates statistics on CPU in `float64`, and SciPy computes the matrix square root. FID uses all real/fake features, while IS uses all generated features. Each metric is computed once after global accumulation; IS has no splits, split standard deviation, or batch-score averaging.

The result directories contain `metrics.json`, `feature_probabilities.pt`, `evaluation_generated_grid.png`, and `evaluation_manifest.json`, including checkpoint/weight hashes, source paths, versions, devices, seeds, preprocessing, and numerical warnings. The two recorded real-probability caches matched elementwise, and both evaluations reported no numerical warnings.

Interpretation is limited by the ImageNet/chest-X-ray domain mismatch, a single evaluation seed, and finite sample size. With `N=624 < D=1000`, each sample covariance has rank at most 623. The code adds no feature projection, FID clamp, or new regularizer and retains Ignite's upstream numerical handling. Lower FID and higher IS favor beta 0.1 under this particular protocol, but neither the scores nor the still-blurred samples establish medical realism.

## Tests

Run the test suite and an optional real-data forward/backward check:

```bash
python -m unittest discover -s tests -v
python model.py --device cpu
```

Tests cover data filtering and split fingerprints, both model resolutions, analytic loss/reduction behavior, gradients, training and checkpoint resume, seeded generation, matched comparisons, preprocessing, global FID/IS accumulation, and final-batch inclusion. Full tests require the metric dependencies.

## Main files

| Path | Purpose |
| --- | --- |
| `dataset.py` | Grayscale preprocessing, filename-group holdout, loaders, manifests |
| `model.py` | Encoder, reparameterization, decoder, scaled BCE + KL loss |
| `train.py` | Training, fixed-noise validation, previews, resumable checkpoints |
| `generate.py` | Prior samples and optional posterior-mean reconstructions |
| `compare_experiments.py` | Matched beta-only comparisons |
| `compare_resolutions.py` | Matched 64/128 comparisons and common-target fidelity |
| `evaluate.py` | Documented Ignite probability-based FID and global IS |
| `config.json`, `configs/` | Baseline and three experiment configurations |
| `tests/` | Reproducibility and numerical behavior checks |

Further references: [PyTorch installation](https://pytorch.org/get-started/locally/), [MPS backend](https://docs.pytorch.org/docs/stable/notes/mps.html), [Ignite FID](https://docs.pytorch.org/ignite/generated/ignite.metrics.FID.html), [Ignite InceptionScore](https://docs.pytorch.org/ignite/generated/ignite.metrics.InceptionScore.html), and [Ignite v0.5.5 Inception wrapper source](https://github.com/pytorch/ignite/blob/v0.5.5/ignite/metrics/gan/utils.py).

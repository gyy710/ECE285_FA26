"""Compare matched 64/128 VAE checkpoints with fixed previews and all-val fidelity.

Run: python compare_resolutions.py --device cpu
Native BCE curves have different targets; common-target MAE/MSE use the same
128-pixel validation targets. Neither establishes clinical or perceptual quality.
"""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F
from torchvision.utils import make_grid

import generate
from dataset import CLASSES, build_dataloaders, save_split_manifest


PROJECT_ROOT = Path(__file__).resolve().parent
INTERPOLATION = "64 to 128: torch bilinear interpolation, align_corners=False; interpolation does not restore missing 64-pixel detail."


def _project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _scale(config):
    return config["training"].get("reconstruction_scale", 1.0)


def _check_compatibility(baseline, experiment):
    """Only native size and its reconstruction normalization may change."""
    if baseline["epoch"] != experiment["epoch"]:
        raise ValueError("Checkpoint epoch values differ; compare the same actual epoch.")
    if not baseline.get("split_fingerprint") or baseline["split_fingerprint"] != experiment.get("split_fingerprint"):
        raise ValueError("Checkpoint split fingerprint values differ or are missing.")
    checkpoints = (baseline, experiment)
    sizes = [checkpoint["config"]["data"]["image_size"] for checkpoint in checkpoints]
    if sizes != [64, 128]:
        raise ValueError("Resolution comparison requires baseline 64 and experiment 128.")
    scales = [_scale(checkpoint["config"]) for checkpoint in checkpoints]
    if any(not math.isfinite(scale) or scale <= 0 for scale in scales):
        raise ValueError("Reconstruction scale must be finite and positive.")
    references = [scale * size ** 2 for scale, size in zip(scales, sizes)]
    if not math.isclose(references[0], references[1], rel_tol=1e-12):
        raise ValueError("Reconstruction reference pixel normalization differs (scale * image_size^2).")
    comparable = []
    for checkpoint in checkpoints:
        config = copy.deepcopy(checkpoint["config"])
        config["data"].pop("image_size")
        for field in ("reconstruction_scale", "epochs", "output_dir"):
            config["training"].pop(field, None)
        comparable.append(config)
        epoch = checkpoint["epoch"]
        history = checkpoint.get("history", [])
        if not isinstance(epoch, int) or epoch < 1 or [row.get("epoch") for row in history] != list(range(1, epoch + 1)):
            raise ValueError("Checkpoint history must contain contiguous epochs 1 through its saved epoch.")
        for row in history:
            for field in ("val_reconstruction", "val_kl"):
                if field not in row:
                    raise ValueError(f"Checkpoint history is missing {field}.")
            for field, value in row.items():
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"Checkpoint history contains a non-finite or nonnumeric {field}.")
    if comparable[0] != comparable[1]:
        raise ValueError("Checkpoint config differs outside data.image_size and training.reconstruction_scale/epochs/output_dir.")


def _check_validation_sources(loaders):
    datasets = [loader.dataset for loader in loaders]
    if datasets[0].classes != datasets[1].classes or datasets[0].samples != datasets[1].samples:
        raise ValueError("Validation source paths / label order differ between native resolutions.")
    if not len(datasets[0]):
        raise ValueError("Validation source split is empty.")


def _upsample(images):
    return F.interpolate(images, size=(128, 128), mode="bilinear", align_corners=False)


@torch.no_grad()
def evaluate_common_target(models, validation_loaders, device):
    """Evaluate every matched validation image, with pixel-weighted MAE and MSE.

    Each model receives its native input; both predictions use the same 128 target.
    No test loader is iterated. Summing errors before dividing includes short batches.
    """
    _check_validation_sources(validation_loaders)
    sums = [{"mae": 0.0, "mse": 0.0} for _ in models]
    samples, pixels = 0, 0
    for baseline_batch, experiment_batch in zip(*validation_loaders, strict=True):
        native_inputs = [baseline_batch[0], experiment_batch[0]]
        if not torch.equal(baseline_batch[1], experiment_batch[1]) or native_inputs[0].shape[0] != native_inputs[1].shape[0]:
            raise ValueError("Validation batch label order or sample count differs.")
        for images, size in zip(native_inputs, (64, 128)):
            if tuple(images.shape[1:]) != (1, size, size):
                raise ValueError(f"Expected native validation input [B,1,{size},{size}].")
        target = native_inputs[1].double()
        predictions = [generate.reconstruct_samples(model, images, device) for model, images in zip(models, native_inputs)]
        predictions[0] = _upsample(predictions[0])
        for totals, prediction in zip(sums, predictions):
            if prediction.shape != target.shape or not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
                raise ValueError("Common-target prediction shape differs or pixels are non-finite.")
            error = prediction.double() - target
            totals["mae"] += error.abs().sum().item()
            totals["mse"] += error.square().sum().item()
        samples += target.shape[0]
        pixels += target.numel()
    if samples != len(validation_loaders[0].dataset):
        raise ValueError("Common-target evaluation must include all validation samples; do not drop the last batch.")
    metrics = [{metric: total / pixels for metric, total in totals.items()} for totals in sums]
    relative = {metric: 100 * (metrics[1][metric] - metrics[0][metric]) / metrics[0][metric]
                if metrics[0][metric] else None for metric in ("mae", "mse")}
    return {
        "sample_count": samples, "pixel_count": pixels, "target_image_size": 128,
        "baseline": metrics[0], "experiment": metrics[1], "relative_change_percent": relative,
        "relative_change_formula": "100 * (experiment - baseline) / baseline; negative means lower error; null if baseline is zero.",
        "method": "Deterministic posterior mean reconstructions of all matched validation images; baseline bilinear upsampled to the same real 128 target; pixel-weighted MAE/MSE.",
        "interpretation": "Same-target fidelity evidence; not a clinical metric or proof of perceptual improvement.",
    }


def _save_reconstructions(targets, reconstructions, sources, labels, destination):
    fig, axes = plt.subplots(len(targets), 3, figsize=(10, 2.5 * len(targets)), squeeze=False)
    for column, (images, title) in enumerate(zip((targets, *reconstructions), ("Real target (128×128)", *labels))):
        axes[0, column].set_title(title, fontsize=10)
        for row, image in enumerate(images):
            axis = axes[row, column]
            axis.imshow(image[0].numpy(), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            axis.set_xticks([])
            axis.set_yticks([])
            if column == 0:
                axis.set_ylabel(Path(sources[row]).parent.name)
    fig.suptitle("Fixed validation filenames · posterior mean reconstruction", fontsize=12)
    fig.text(.5, .015, "64→128 uses bilinear interpolation; resizing cannot restore missing detail.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .95))
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def _save_generation(samples, labels, seed, destination):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for axis, images, label in zip(axes, samples, labels):
        grid = make_grid(images, nrow=8, padding=2, pad_value=1)[0].numpy()
        axis.imshow(grid, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axis.set_title(label, fontsize=11)
        axis.set_axis_off()
    fig.suptitle(f"64 prior samples per model · same latent draws · seed {seed} · nearest display")
    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def _save_curves(checkpoints, labels, destination):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for checkpoint, label in zip(checkpoints, labels):
        history = checkpoint["history"]
        reference = _scale(checkpoint["config"]) * checkpoint["config"]["data"]["image_size"] ** 2
        epochs = [row["epoch"] for row in history]
        axes[0].plot(epochs, [row["val_reconstruction"] / reference for row in history], label=label)
        axes[1].plot(epochs, [row["val_kl"] for row in history], label=label)
    for axis, title, ylabel in zip(axes, ("Validation BCE per native input pixel", "Validation KL"), ("Scaled reconstruction / (scale × size²)", "Mean per image; sum over latent dimensions")):
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=.25)
        axis.legend(fontsize=8)
    fig.suptitle("Recorded fixed-noise Monte Carlo validation")
    fig.text(.5, .015, "Targets differ in resolution: native reconstruction metrics alone do not establish superiority.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .95))
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path("outputs/beta_0p1/last.pt"))
    parser.add_argument("--experiment", type=Path, default=Path("outputs/resolution_128/last.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/comparison_resolution"))
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    args = parser.parse_args(argv)
    paths = [_project_path(args.baseline), _project_path(args.experiment)]
    output_dir = _project_path(args.output_dir)
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f"Output directory is not empty: {output_dir}; choose a new --output-dir.")
    device = torch.device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; use --device cpu.")
    loaded = [generate.load_trained_model(path, device) for path in paths]
    models, checkpoints = zip(*loaded)
    _check_compatibility(*checkpoints)
    loaders, manifests = [], []
    for checkpoint in checkpoints:
        config = checkpoint["config"]
        current_loaders = build_dataloaders(config)
        # Manifest scans memberships only, including test filenames, never test pixels.
        with tempfile.TemporaryDirectory() as temporary:
            manifest = save_split_manifest(current_loaders, config, Path(temporary) / "manifest.json")
        if manifest["fingerprint"] != checkpoint["split_fingerprint"]:
            raise ValueError("Actual dataset split fingerprint does not match its checkpoint.")
        loaders.append(current_loaders)
        manifests.append(manifest)
    val_loaders = [item["val"] for item in loaders]
    _check_validation_sources(val_loaders)
    common_target = evaluate_common_target(models, val_loaders, device)
    datasets = [loader.dataset for loader in val_loaders]
    indices = []
    for name in CLASSES:
        target = datasets[0].class_to_idx[name]
        candidates = [index for index, (_, label) in enumerate(datasets[0].samples) if label == target]
        indices.extend(sorted(candidates, key=lambda index: datasets[0].samples[index][0])[:2])
    originals = [torch.stack([dataset[index][0] for index in indices]) for dataset in datasets]
    reconstructions = [generate.reconstruct_samples(model, images, device) for model, images in zip(models, originals)]
    reconstructions[0] = _upsample(reconstructions[0])
    root = _project_path(checkpoints[0]["config"]["data"]["root"])
    sources = [str(Path(datasets[0].samples[index][0]).relative_to(root)) for index in indices]
    seed = checkpoints[0]["config"]["generation"]["seed"]
    samples = [generate.generate_samples(model, 64, seed, 64, device) for model in models]
    epoch = checkpoints[0]["epoch"]
    labels = [f"{role}: native {checkpoint['config']['data']['image_size']}×{checkpoint['config']['data']['image_size']}\nβ={checkpoint['config']['training']['beta']:g} · epoch {epoch}"
              for role, checkpoint in zip(("Baseline", "Experiment"), checkpoints)]
    summary = {
        "epoch": epoch, "device": str(device),
        "generation": {"seed": seed, "count_per_model": 64, "prior": "standard_normal_N(0,I)", "display_interpolation": "nearest; same physical display size; no sharpening"},
        "split": {key: manifests[0][key] for key in ("fingerprint", "protocol", "counts", "class_counts", "splits")},
        "current_membership_verified_for_both_configs": True,
        "reconstruction_source_files": sources,
        "reconstruction_selection": "First two filenames per class in NORMAL then PNEUMONIA order; fewer if unavailable.",
        "reconstruction_method": "Deterministic posterior mean: decode(mu).sigmoid(); each model receives its native input size.",
        "interpolation_method": INTERPOLATION,
        "validation_metric_method": "Recorded fixed-noise Monte Carlo validation, mean per image; distinct from posterior mean previews and the separately recomputed all-validation common-target evaluation.",
        "native_metric_note": "Native BCE per pixel = recorded scaled reconstruction / (reconstruction_scale * image_size^2). Targets differ in resolution, so native metrics alone do not establish superiority.",
        "capacity_caveat": "Resolution plus necessary dimension/capacity adaptation: four convolution stages end at 4×4 versus 8×8 and linear dimensions/parameter counts differ. This is not a pure pixels-only ablation.",
        "total_loss_note": "No cross-resolution total-loss ranking: reconstruction targets and native dimensions differ.",
        "automatic_quality_winner_selection": False,
        "common_target_validation": common_target,
    }
    for role, model, checkpoint, path, images in zip(("baseline", "experiment"), models, checkpoints, paths, samples):
        config = checkpoint["config"]
        size, scale = config["data"]["image_size"], _scale(config)
        last = checkpoint["history"][-1]
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        summary[role] = {
            "checkpoint": str(path), "checkpoint_sha256": digest, "checkpoint_epoch": checkpoint["epoch"],
            "config": config, "beta": config["training"]["beta"], "image_size": size,
            "reconstruction_scale": scale, "reference_pixels": scale * size ** 2,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "generation_tensor_shape": list(images.shape),
            "validation": {"scaled_reconstruction": last["val_reconstruction"], "kl": last["val_kl"],
                           "reconstruction_per_native_pixel": last["val_reconstruction"] / (scale * size ** 2)},
        }
    # Finish compatibility, provenance and inference before writing requested outputs.
    serialized = json.dumps(summary, indent=2, allow_nan=False) + "\n"
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_reconstructions(originals[1], reconstructions, sources, labels, output_dir / "reconstruction_comparison.png")
    _save_generation(samples, labels, seed, output_dir / "generation_comparison.png")
    _save_curves(checkpoints, labels, output_dir / "validation_curves.png")
    (output_dir / "summary.json").write_text(serialized)
    print(f"Saved matched epoch {epoch} resolution comparison to {output_dir}")
    print(f"All-validation common-target samples: {common_target['sample_count']}; MAE/MSE changes (%): {common_target['relative_change_percent']}")
    print("No automatic image-quality winner; inspect fixed previews and fidelity evidence together.")
    return summary


if __name__ == "__main__":
    main()

"""Compare a beta-only VAE experiment using matched checkpoints and fixed examples.

Run: python compare_experiments.py --device cpu
Exports validation reconstructions, seeded prior grids, validation curves and provenance.
Validation loss and qualitative image quality are different pieces of evidence.
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
from torchvision.utils import make_grid

import generate
from dataset import CLASSES, build_dataloaders, save_split_manifest


PROJECT_ROOT = Path(__file__).resolve().parent


def _project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _check_compatibility(baseline, experiment):
    """Require one changed training variable, and identical recorded epoch ranges."""
    if baseline["epoch"] != experiment["epoch"]:
        raise ValueError("Checkpoint epoch values differ; compare the same epoch.")
    fingerprint = baseline.get("split_fingerprint")
    if not fingerprint or fingerprint != experiment.get("split_fingerprint"):
        raise ValueError("Checkpoint split fingerprint values differ or are missing.")
    comparable = []
    for checkpoint in (baseline, experiment):
        config = copy.deepcopy(checkpoint["config"])
        for field in ("beta", "epochs", "output_dir"):
            config["training"].pop(field, None)
        comparable.append(config)
        history = checkpoint.get("history", [])
        epochs = [row["epoch"] for row in history]
        if not epochs or epochs != list(range(1, checkpoint["epoch"] + 1)):
            raise ValueError("Checkpoint history must record each epoch through its saved epoch.")
        for row in history:
            for field in ("val_reconstruction", "val_kl"):
                if not math.isfinite(row[field]):
                    raise ValueError(f"Checkpoint history contains a non-finite {field}.")
    if comparable[0] != comparable[1]:
        raise ValueError("Checkpoint config differs outside training.beta/epochs/output_dir.")
    betas = [checkpoint["config"]["training"]["beta"] for checkpoint in (baseline, experiment)]
    if any(not math.isfinite(beta) or beta < 0 for beta in betas) or betas[0] == betas[1]:
        raise ValueError("A beta comparison requires different finite nonnegative beta values.")


def _save_reconstructions(originals, reconstructions, sources, labels, destination):
    fig, axes = plt.subplots(len(originals), 3, figsize=(9, 2.5 * len(originals)), squeeze=False)
    columns = (originals, *reconstructions)
    height, width = originals.shape[-2:]
    for column, title in enumerate((f"Real ({height}×{width})", *labels)):
        axes[0, column].set_title(title, fontsize=11)
        for row in range(len(originals)):
            axis = axes[row, column]
            axis.imshow(columns[column][row, 0].numpy(), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            axis.set_xticks([])
            axis.set_yticks([])
            if column == 0:
                axis.set_ylabel(Path(sources[row]).parent.name, fontsize=10)
    fig.suptitle("Fixed validation examples · posterior mean reconstruction", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def _save_generation(samples, labels, seed, destination):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for axis, images, label in zip(axes, samples, labels):
        grid = make_grid(images, nrow=8, padding=2, pad_value=1)[0].numpy()
        axis.imshow(grid, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axis.set_title(label)
        axis.set_axis_off()
    fig.suptitle(f"64 standard normal prior samples per model · matching latent draws · seed {seed}")
    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def _save_curves(checkpoints, labels, destination):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, metric, title in zip(axes, ("reconstruction", "kl"), ("Validation reconstruction loss", "Validation KL")):
        for checkpoint, label in zip(checkpoints, labels):
            history = checkpoint["history"]
            axis.plot([row["epoch"] for row in history], [row[f"val_{metric}"] for row in history], label=label)
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Scaled pixel BCE sum per image" if metric == "reconstruction" else "KL sum per image")
        axis.legend(fontsize=8)
        axis.grid(alpha=0.25)
    fig.suptitle("Recorded training validation metrics · fixed-noise Monte Carlo")
    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path("outputs/baseline/last.pt"))
    parser.add_argument("--experiment", type=Path, default=Path("outputs/beta_0p1/last.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/comparison_beta"))
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
    baseline, experiment = checkpoints
    _check_compatibility(baseline, experiment)
    config = baseline["config"]
    loaders = build_dataloaders(config)
    with tempfile.TemporaryDirectory() as temporary:
        manifest = save_split_manifest(loaders, config, Path(temporary) / "split_manifest.json")
    if manifest["fingerprint"] != baseline["split_fingerprint"]:
        raise ValueError("Actual dataset split fingerprint does not match the checkpoints.")

    # Only validation images are decoded. The manifest reads test filenames, never test pixels.
    validation = loaders["val"].dataset
    indices = []
    for class_name in CLASSES:
        target = validation.class_to_idx[class_name]
        candidates = [index for index, (_, label) in enumerate(validation.samples) if label == target]
        indices.extend(sorted(candidates, key=lambda index: validation.samples[index][0])[:2])
    if not indices:
        raise ValueError("Validation split has no reconstruction examples.")
    originals = torch.stack([validation[index][0] for index in indices])
    data_root = _project_path(config["data"]["root"])
    sources = [str(Path(validation.samples[index][0]).relative_to(data_root)) for index in indices]
    reconstructions = [generate.reconstruct_samples(model, originals, device) for model in models]
    seed = config["generation"]["seed"]
    samples = [generate.generate_samples(model, 64, seed, 64, device) for model in models]
    epoch = baseline["epoch"]
    labels = [f"{role}: β={checkpoint['config']['training']['beta']:g} · epoch {epoch}"
              for role, checkpoint in zip(("Baseline", "Experiment"), checkpoints)]

    summary = {
        "epoch": epoch,
        "device": str(device),
        "generation": {"seed": seed, "count_per_model": 64, "prior": "standard_normal_N(0,I)"},
        "split": {"fingerprint": manifest["fingerprint"], "counts": manifest["counts"],
                  "class_counts": manifest["class_counts"], "protocol": manifest["protocol"]},
        "reconstruction_source_files": sources,
        "reconstruction_selection": "First two filenames per class in NORMAL, PNEUMONIA order; fewer if unavailable.",
        "reconstruction_method": "Deterministic posterior mean: decode(mu).sigmoid().",
        "validation_metric_method": "Recorded training fixed-noise Monte Carlo validation metrics, mean per image; reconstruction is the scaled pixel BCE sum (training.reconstruction_scale, default 1.0), KL is unscaled; not recomputed by this script.",
        "total_loss_note": "Cross-beta total losses are incomparable because beta changes the KL weight; they do not establish image-quality superiority.",
        "automatic_quality_winner_selection": False,
        "quality_note": "These metrics and fixed qualitative previews do not automatically establish improved image quality or medical quality. No FID/IS is computed.",
        "reconstruction_loss_percent_change_formula": "100 * (experiment - baseline) / baseline; negative means lower recorded reconstruction loss.",
    }
    for role, checkpoint, path in zip(("baseline", "experiment"), checkpoints, paths):
        last = checkpoint["history"][-1]
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        summary[role] = {
            "checkpoint": str(path), "checkpoint_sha256": digest,
            "beta": checkpoint["config"]["training"]["beta"],
            "image_size": checkpoint["config"]["data"]["image_size"],
            "reconstruction_scale": checkpoint["config"]["training"].get("reconstruction_scale", 1.0),
            "validation": {"reconstruction": last["val_reconstruction"], "kl": last["val_kl"]},
        }
    base_loss = summary["baseline"]["validation"]["reconstruction"]
    experiment_loss = summary["experiment"]["validation"]["reconstruction"]
    summary["reconstruction_loss_percent_change"] = 100 * (experiment_loss - base_loss) / base_loss if base_loss else None

    # Checks and inference finish before any output is written to the requested directory.
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_reconstructions(originals, reconstructions, sources, labels, output_dir / "reconstruction_comparison.png")
    _save_generation(samples, labels, seed, output_dir / "generation_comparison.png")
    _save_curves(checkpoints, labels, output_dir / "validation_curves.png")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"Saved matched epoch {epoch} comparison to {output_dir}")
    print(f"Recorded validation reconstruction loss change: {summary['reconstruction_loss_percent_change']}%")
    print("Image quality requires inspection; no winner is selected automatically.")
    return summary


if __name__ == "__main__":
    main()

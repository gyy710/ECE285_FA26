"""Step 4: Train the convolutional VAE and save losses, previews, and resumable checkpoints.

Example: python train.py --epochs 1 --output-dir outputs/smoke
Resume: python train.py --epochs 30 --resume outputs/smoke/last.pt
--epochs always specifies the total target number of epochs, not additional epochs.
"""

import argparse
import copy
import csv
import json
import platform
import random
import sys
import tempfile
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save images without a graphical desktop; must precede pyplot.
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torchvision.utils import save_image

from model import build_model, vae_loss


PROJECT_ROOT = Path(__file__).resolve().parent
METRICS = ("total", "reconstruction", "kl")


def _project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def run_epoch(model, loader, device, beta, optimizer=None, validation_seed=12345, reconstruction_scale=1.0):
    """Return epoch means weighted by actual sample counts; optimizer=None means validation.

    reconstruction_scale scales only the pixel-sum term; the default of 1 preserves earlier experiments.
    Each validation run resets a separate CPU generator so each image uses the same posterior noise.
    Validation therefore estimates the sampled weighted objective without changing the global torch RNG
    used for training.
    """
    training = optimizer is not None
    model.train(training)
    noise_generator = torch.Generator(device="cpu").manual_seed(validation_seed)
    sums = dict.fromkeys(METRICS, 0.0)
    samples = 0
    with torch.set_grad_enabled(training):
        for batch in loader:
            images = batch[0].to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
                logits, mu, logvar = model(images)
            else:
                mu, logvar = model.encode(images)
                noise = torch.randn(mu.shape, generator=noise_generator, dtype=mu.dtype).to(device)
                z = mu + torch.exp(0.5 * logvar) * noise
                logits = model.decode(z)
            losses = vae_loss(
                logits, images, mu, logvar, beta=beta, reconstruction_scale=reconstruction_scale
            )
            if not all(torch.isfinite(loss).item() for loss in losses):
                raise RuntimeError("Non-finite VAE loss; training stopped before an optimizer update.")
            if training:
                losses[0].backward()
                optimizer.step()
            # vae_loss returns batch means; the last batch may be shorter, so do not average batches equally.
            batch_size = images.shape[0]
            for name, loss in zip(METRICS, losses):
                sums[name] += loss.item() * batch_size
            samples += batch_size
    if samples == 0:
        raise ValueError("Cannot run an epoch on an empty loader (zero samples).")
    return {**{name: value / samples for name, value in sums.items()}, "samples": samples}


def save_previews(model, validation_dataset, device, config, output_dir, epoch):
    """Use fixed prior z; show real images in row one and sigmoid-decoded posterior means in row two.

    Read the validation dataset directly so previews do not advance any DataLoader generator.
    """
    model.eval()
    generator = torch.Generator(device="cpu").manual_seed(config.get("generation", {}).get("seed", 123))
    count = min(64, config.get("generation", {}).get("grid_samples", 64))
    z = torch.randn(count, config["model"]["latent_dim"], generator=generator).to(device)
    images = torch.stack([validation_dataset[i][0] for i in range(min(8, len(validation_dataset)))])
    with torch.no_grad():
        generated = model.decode(z).sigmoid().cpu()
        images = images.to(device)
        mu, _ = model.encode(images)
        reconstruction = model.decode(mu).sigmoid()
        comparison = torch.cat((images, reconstruction)).cpu()
    output_dir = Path(output_dir)
    save_image(generated, output_dir / f"generated_epoch_{epoch:03d}.png", nrow=8)
    save_image(comparison, output_dir / f"reconstruction_epoch_{epoch:03d}.png", nrow=len(images))


def _save_history(history, output_dir):
    with (output_dir / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    epochs = [row["epoch"] for row in history]
    for axis, metric in zip(axes, METRICS):
        for split in ("train", "val"):
            axis.plot(epochs, [row[f"{split}_{metric}"] for row in history], marker="o", label=split)
        axis.set(title=metric.capitalize(), xlabel="Epoch", ylabel="Loss per image")
        axis.legend()
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "loss_curves.png", dpi=120)
    plt.close(figure)


def _cpu_copy(value):
    """Create an independent CPU snapshot so later training cannot change best parameters; retain safe-load types."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return copy.deepcopy(value)


def _rng_states(loaders, device):
    numpy_state = np.random.get_state()
    states = {
        "torch_cpu": torch.get_rng_state(),
        "python": random.getstate(),
        # NumPy ndarrays cannot be safely loaded with torch.load(weights_only=True).
        "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
        "loaders": {name: loader.generator.get_state() for name, loader in loaders.items()},
    }
    if device.type == "mps":
        states["torch_mps"] = torch.mps.get_rng_state().cpu()
    return states


def _restore_rng(states, loaders, device):
    torch.set_rng_state(states["torch_cpu"])
    random.setstate(states["python"])
    numpy_state = states["numpy"]
    np.random.set_state((numpy_state[0], np.array(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
    for name, loader in loaders.items():
        loader.generator.set_state(states["loaders"][name])
    if device.type == "mps" and "torch_mps" in states:
        torch.mps.set_rng_state(states["torch_mps"])


def _compatible_config(config):
    """Allow changes to total epochs and output directory; model, optimization, and data settings must match."""
    training = {key: value for key, value in config["training"].items()
                if key not in ("epochs", "output_dir")}
    return {"seed": config["seed"], "data": config["data"], "model": config["model"],
            "training": training, "generation": config.get("generation", {})}


def _sync(device):
    if device.type == "mps":
        torch.mps.synchronize()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.json")
    parser.add_argument("--epochs", type=int, help="Total target number of epochs, including restored epochs.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "mps"))
    parser.add_argument("--resume", type=Path, help="Resume from best.pt or last.pt.")
    args = parser.parse_args(argv)
    config = json.loads(_project_path(args.config).read_text())
    training = config["training"]
    if args.epochs is not None:
        training["epochs"] = args.epochs
    if args.device:
        training["device"] = args.device
    if args.output_dir is not None:
        training["output_dir"] = str(args.output_dir)
    if training["epochs"] < 1:
        parser.error("--epochs must be positive.")
    if training["batch_size"] < 1:
        raise ValueError("training.batch_size must be positive.")
    if training["optimizer"].lower() != "adam":
        raise ValueError("This trainer supports training.optimizer='adam'.")
    if training.get("reconstruction_loss", "bce_with_logits") != "bce_with_logits":
        raise ValueError("This VAE uses reconstruction_loss='bce_with_logits'.")
    if not np.isfinite(training["beta"]) or training["beta"] < 0:
        raise ValueError("training.beta must be finite and nonnegative.")
    reconstruction_scale = training.get("reconstruction_scale", 1.0)
    if not np.isfinite(reconstruction_scale) or reconstruction_scale <= 0:
        raise ValueError("training.reconstruction_scale must be finite and positive.")
    if config.get("generation", {}).get("grid_samples", 64) < 1:
        raise ValueError("generation.grid_samples must be positive.")
    device = torch.device(training["device"])
    if device.type not in ("cpu", "mps"):
        raise ValueError("Choose training.device='cpu' or 'mps'.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; check the environment or use --device cpu.")

    output_dir = _project_path(training["output_dir"])
    training["output_dir"] = str(output_dir)
    # real_samples.png / split_manifest.json from data preparation may already exist.
    artifacts = ("config.json", "environment.json", "history.csv", "loss_curves.png", "best.pt", "last.pt")
    if not args.resume and (any((output_dir / name).exists() for name in artifacts)
                            or list(output_dir.glob("*_epoch_*.png"))):
        raise FileExistsError(f"Training artifacts already exist in {output_dir}; use --resume or a new output directory.")
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(_project_path(args.resume), map_location="cpu", weights_only=True)
        if _compatible_config(config) != _compatible_config(checkpoint["config"]):
            raise ValueError("Resume configuration is incompatible with saved model/training/data configuration.")
        if training["epochs"] <= checkpoint["epoch"]:
            raise ValueError("No remaining epochs: --epochs must exceed the checkpoint epoch.")

    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    from dataset import build_dataloaders, save_split_manifest
    loaders = build_dataloaders(config)
    # Compare splits before writing to the run directory so an incompatible resume cannot overwrite files.
    with tempfile.TemporaryDirectory() as temporary:
        manifest = save_split_manifest(loaders, config, Path(temporary) / "split_manifest.json")
    if checkpoint and checkpoint["split_fingerprint"] != manifest["fingerprint"]:
        raise ValueError("Resume split fingerprint differs: dataset membership or validation split changed.")
    if any(len(loaders[split].dataset) == 0 for split in ("train", "val")):
        raise ValueError("Training and validation datasets must contain samples.")
    model = build_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=training["learning_rate"])
    history, start_epoch, best_val_loss, best_checkpoint = [], 1, float("inf"), None
    if checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        history = copy.deepcopy(checkpoint["history"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]
        best_checkpoint = checkpoint.get("best_checkpoint", checkpoint)
        # Restore RNG after model/loader initialization so initialization cannot affect the next epoch's random sequence.
        _restore_rng(checkpoint["rng_states"], loaders, device)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    environment = {"python": sys.version, "torch": str(torch.__version__),
                   "torchvision": str(torchvision.__version__), "device": str(device),
                   "platform": platform.platform(), "machine": platform.machine(),
                   "mps_available": torch.backends.mps.is_available()}
    (output_dir / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    if best_checkpoint is not None:
        torch.save(best_checkpoint, output_dir / "best.pt")
    print(f"Device: {device}; split sizes: {manifest['counts']}; target epochs: {training['epochs']}; "
          f"reconstruction scale: {reconstruction_scale}")
    for epoch in range(start_epoch, training["epochs"] + 1):
        _sync(device)
        started = time.perf_counter()
        train_metrics = run_epoch(
            model, loaders["train"], device, training["beta"], optimizer=optimizer,
            reconstruction_scale=reconstruction_scale,
        )
        val_metrics = run_epoch(model, loaders["val"], device, training["beta"],
                                validation_seed=training.get("validation_seed", config["seed"] + 1),
                                reconstruction_scale=reconstruction_scale)
        save_previews(model, loaders["val"].dataset, device, config, output_dir, epoch)
        _sync(device)
        elapsed = time.perf_counter() - started
        row = {"epoch": epoch}
        row.update({f"{split}_{name}": value for split, metrics in
                    (("train", train_metrics), ("val", val_metrics)) for name, value in metrics.items()})
        row["elapsed_seconds"] = elapsed
        history.append(row)
        improved = val_metrics["total"] < best_val_loss
        best_val_loss = min(best_val_loss, val_metrics["total"])
        snapshot = {
            "epoch": epoch, "model_state_dict": _cpu_copy(model.state_dict()),
            "optimizer_state_dict": _cpu_copy(optimizer.state_dict()), "config": copy.deepcopy(config),
            "best_val_loss": best_val_loss, "history": copy.deepcopy(history),
            "split_fingerprint": manifest["fingerprint"], "rng_states": _rng_states(loaders, device),
        }
        if improved:
            best_checkpoint = snapshot
            torch.save(best_checkpoint, output_dir / "best.pt")
        # Embed an independent best snapshot so moving last.pt alone still preserves the true best model on resume.
        torch.save({**snapshot, "best_checkpoint": best_checkpoint}, output_dir / "last.pt")
        _save_history(history, output_dir)
        print(f"Epoch {epoch}/{training['epochs']} | "
              f"train total={train_metrics['total']:.3f} recon={train_metrics['reconstruction']:.3f} "
              f"KL={train_metrics['kl']:.3f} n={train_metrics['samples']} | "
              f"val total={val_metrics['total']:.3f} recon={val_metrics['reconstruction']:.3f} "
              f"KL={val_metrics['kl']:.3f} n={val_metrics['samples']} | {elapsed:.2f}s", flush=True)
    return history


if __name__ == "__main__":
    main()

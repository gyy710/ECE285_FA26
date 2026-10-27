"""Step 5: Generate images from the trained VAE prior, with optional validation reconstructions saved separately.

Example: python generate.py --reconstruct
By default, read outputs/baseline/best.pt and write results to generation/ in the same directory.
Random generation does not require the original data; only --reconstruct loads validation images from the same split.
"""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

from PIL import Image
import torch
from torchvision.utils import save_image

from model import build_model


PROJECT_ROOT = Path(__file__).resolve().parent


def _project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_trained_model(checkpoint_path, device):
    """Use the checkpoint's configuration rather than the potentially modified root config.json."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = build_model(checkpoint["config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval(), checkpoint


@torch.no_grad()
def generate_samples(model, num_samples, seed, batch_size, device):
    """Sample from N(0,I) and decode, returning a CPU tensor of shape [N,1,H,W] in [0,1].

    Draw all z values at once with a separate CPU generator, then decode in batches.
    Changing batch_size therefore does not change the latent variables or advance
    the global random state used for training. Floating-point computations on
    different devices or with different batches may still produce small numerical differences.
    """
    if num_samples < 1 or batch_size < 1:
        raise ValueError("num_samples and batch_size must be positive.")
    model.eval()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    dtype = next(model.parameters()).dtype
    latent = torch.randn(num_samples, model.latent_dim, generator=generator, dtype=dtype)
    batches = []
    for start in range(0, num_samples, batch_size):
        logits = model.decode(latent[start:start + batch_size].to(device))
        images = logits.sigmoid().cpu()
        if not torch.isfinite(images).all().item():
            raise RuntimeError("Generated images contain non-finite values.")
        batches.append(images)
    return torch.cat(batches)


@torch.no_grad()
def reconstruct_samples(model, images, device):
    """Reconstruct with the posterior mean for stable comparisons; these images are not randomly generated samples."""
    model.eval()
    mu, _ = model.encode(images.to(device))
    reconstructed = model.decode(mu).sigmoid().cpu()
    if not torch.isfinite(reconstructed).all().item():
        raise RuntimeError("Reconstructed images contain non-finite values.")
    return reconstructed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "outputs/baseline/best.pt")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--num-samples", type=int, help="Defaults to grid_samples from the checkpoint.")
    parser.add_argument("--seed", type=int, help="Defaults to generation.seed from the checkpoint.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("cpu", "mps"))
    parser.add_argument("--reconstruct", action="store_true", help="Also save validation images and a posterior-mean reconstruction comparison.")
    args = parser.parse_args(argv)
    checkpoint_path = _project_path(args.checkpoint)
    output_dir = _project_path(args.output_dir) if args.output_dir else checkpoint_path.parent / "generation"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}; choose a new --output-dir.")
    device = torch.device(args.device or ("mps" if torch.backends.mps.is_available() else "cpu"))
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; use --device cpu.")
    model, checkpoint = load_trained_model(checkpoint_path, device)
    config = checkpoint["config"]
    generation = config.get("generation", {})
    count = args.num_samples if args.num_samples is not None else generation.get("grid_samples", 64)
    seed = args.seed if args.seed is not None else generation.get("seed", 123)
    images = generate_samples(model, count, seed, args.batch_size, device)

    originals, reconstruction, source_files = None, None, []
    if args.reconstruct:
        from dataset import build_dataloaders, save_split_manifest

        loaders = build_dataloaders(config)
        with tempfile.TemporaryDirectory() as temporary:
            manifest = save_split_manifest(loaders, config, Path(temporary) / "split_manifest.json")
        if manifest["fingerprint"] != checkpoint["split_fingerprint"]:
            raise ValueError("Reconstruction split fingerprint does not match the training checkpoint.")
        dataset = loaders["val"].dataset
        count_reconstructed = min(8, len(dataset))
        originals = torch.stack([dataset[index][0] for index in range(count_reconstructed)])
        reconstruction = reconstruct_samples(model, originals, device)
        data_root = _project_path(config["data"]["root"])
        source_files = [str(Path(path).relative_to(data_root)) for path, _ in dataset.samples[:count_reconstructed]]

    # Write only after all checks pass to avoid partial exports if the data split is incorrect.
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    for index, image in enumerate(images, start=1):
        pixels = image[0].clamp(0, 1).mul(255).round().to(torch.uint8).numpy()
        Image.fromarray(pixels).save(samples_dir / f"sample_{index:04d}.png")
    save_image(images[:64], output_dir / "generated_grid.png", nrow=8)
    if reconstruction is not None:
        save_image(torch.cat((originals, reconstruction)), output_dir / "reconstruction_grid.png", nrow=len(originals))
    with checkpoint_path.open("rb") as handle:
        checkpoint_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    metadata = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_split_fingerprint": checkpoint.get("split_fingerprint"),
        "seed": seed,
        "num_samples": count,
        "batch_size": args.batch_size,
        "grid_samples": min(64, count),
        "latent_dim": model.latent_dim,
        "prior": "standard_normal_N(0,I)",
        "tensor_shape": list(images.shape),
        "tensor_dtype": str(images.dtype),
        "pixel_range_before_png": [images.min().item(), images.max().item()],
        "individual_png_encoding": "8-bit grayscale L; clamp [0,1], multiply by 255, round",
        "device": str(device),
        "torch": str(torch.__version__),
        "reconstruction_split": "val" if args.reconstruct else None,
        "reconstruction_method": "decode(posterior_mean).sigmoid()" if args.reconstruct else None,
        "reconstruction_source_files": source_files,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Checkpoint epoch: {checkpoint['epoch']}; device: {device}; seed: {seed}")
    print(f"Generated {count} prior samples: {tuple(images.shape)}")
    print(f"Exported samples, grid and metadata: {output_dir}")
    if args.reconstruct:
        print(f"Saved separate deterministic reconstruction comparison for {len(source_files)} validation images.")
    return metadata


if __name__ == "__main__":
    main()

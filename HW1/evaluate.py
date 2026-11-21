"""Evaluate the fixed checkpoint on ALL test images with Ignite FID and global IS.

The assignment's default Ignite extractor returns 1000 ImageNet probabilities.
This is not the conventional 2048-dimensional pool3 FID. Feature inference may
use MPS; Ignite covariance accumulation and SciPy sqrtm always run on CPU.
"""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import platform
import tempfile
import time
import warnings

import ignite
from ignite.metrics import FID, InceptionScore
from ignite.metrics.gan.utils import InceptionModel
import numpy as np
import PIL
from PIL import Image
import scipy
import torch
from torch import nn
import torchvision
from torchvision import transforms
from torchvision.utils import save_image

from dataset import build_dataloaders, save_split_manifest
from generate import generate_samples, load_trained_model


PROJECT_ROOT = Path(__file__).resolve().parent
FEATURE_DIMENSION = 1000
IMAGENET_MEAN = (.485, .456, .406)
IMAGENET_STD = (.229, .224, .225)


def _project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def preprocess_images(images):
    """Native grayscale [0,1] -> uint8 PIL -> 299 bilinear RGB -> ImageNet norm.

    ToPILImage intentionally multiplies floats by 255 and truncates to uint8.
    Do not substitute weights.transforms(): its resize/crop geometry differs.
    """
    if (not isinstance(images, torch.Tensor) or images.ndim != 4
            or images.shape[0] == 0 or images.shape[1] != 1
            or min(images.shape[2:]) < 1 or not images.is_floating_point()):
        raise ValueError("Images must be nonempty floating [N,1,H,W] tensors.")
    images = images.detach().cpu()
    if not torch.isfinite(images).all().item() or images.min().item() < 0 or images.max().item() > 1:
        raise ValueError("Images must be finite and in [0,1]; no range rescaling is applied.")
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    return torch.stack([
        normalize(to_tensor(to_pil(image).resize((299, 299), Image.Resampling.BILINEAR).convert("RGB")))
        for image in images
    ])


def _validate_probabilities(probabilities, dimension=None):
    if (not isinstance(probabilities, torch.Tensor) or probabilities.ndim != 2
            or min(probabilities.shape) < 1 or probabilities.dtype != torch.float32):
        raise ValueError("Probabilities must be nonempty float32 [N,D] tensors.")
    if dimension is not None and probabilities.shape[1] != dimension:
        raise ValueError(f"Expected {dimension} probability features; got {probabilities.shape[1]}.")
    if not torch.isfinite(probabilities).all().item() or (probabilities < 0).any().item():
        raise ValueError("Probabilities must be finite and nonnegative.")
    if not torch.allclose(probabilities.sum(dim=1), torch.ones_like(probabilities[:, 0]), rtol=1e-5, atol=1e-6):
        raise ValueError("Probability rows must sum to one; do not provide logits or apply another softmax.")


def create_extractor(device):
    """The sole shared backbone for real, generated, FID, and IS features."""
    extractor = InceptionModel(return_features=False, device=device).eval()
    if not extractor.model.transform_input:
        raise RuntimeError("Selected torchvision Inception wrapper must have transform_input=True.")
    return extractor


@torch.no_grad()
def collect_probabilities(batches, extractor, device, expected_count, dimension=1000, phase="images"):
    """Forward each image once, retaining unmodified CPU float32 probabilities."""
    extractor.eval()
    cached, seen = [], 0
    for batch in batches:
        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        outputs = extractor(preprocess_images(images).to(device)).detach().cpu()
        _validate_probabilities(outputs, dimension)
        if outputs.shape[0] != images.shape[0]:
            raise ValueError("Extractor output count does not match its input batch.")
        seen += outputs.shape[0]
        if seen > expected_count:
            raise ValueError(f"{phase} feature count exceeds expected count {expected_count}.")
        cached.append(outputs)
        print(f"{phase} feature images: {seen}/{expected_count}", flush=True)
    if seen != expected_count or not cached:
        raise ValueError(f"{phase} feature count {seen} differs from expected count {expected_count}.")
    return torch.cat(cached)


def compute_probability_metrics(real, generated, batch_size=32):
    """Accumulate every pair with actual Ignite metrics; compute once globally.

    Identity consumes the cached probabilities using the same semantics as the
    default extractor. Ignite internally converts float32 inputs to CPU float64.
    No jitter, feature projection, FID clamp, or split-averaged IS is introduced.
    """
    _validate_probabilities(real)
    _validate_probabilities(generated)
    if real.shape != generated.shape or real.shape[0] < 2:
        raise ValueError("FID requires equal real/generated feature shapes and at least two samples.")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    real, generated = real.detach().cpu(), generated.detach().cpu()
    dimension = real.shape[1]
    fid = FID(num_features=dimension, feature_extractor=nn.Identity(), device="cpu")
    inception = InceptionScore(num_features=dimension, feature_extractor=nn.Identity(), device="cpu")
    print(f"Accumulating CPU float64 metrics for {len(real)} pairs ({dimension} features)", flush=True)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        for start in range(0, len(real), batch_size):
            fake_batch = generated[start:start + batch_size]
            fid.update((fake_batch, real[start:start + batch_size]))
            inception.update(fake_batch)
        print("Computing global CPU FID covariance/sqrtm and IS (one compute each)...", flush=True)
        try:
            fid_value = float(fid.compute())
            inception_value = float(inception.compute())
        except Exception as error:
            recorded = "; ".join(str(item.message) for item in captured)
            raise RuntimeError(
                f"CPU Ignite FID/IS computation failed with N={len(real)}, D={dimension}; "
                f"rank-deficient covariance may affect sqrtm. Source error: {error}. Warnings: {recorded}"
            ) from error
    if not math.isfinite(fid_value) or not math.isfinite(inception_value):
        raise RuntimeError(f"CPU FID/IS must be finite; got FID={fid_value}, IS={inception_value}.")
    return {"fid": fid_value, "inception_score": inception_value,
            "numerical_warnings": [f"{item.category.__name__}: {item.message}" for item in captured]}


def _validate_config(config):
    evaluation = config.get("evaluation", {})
    for key, required in (("real_split", "test"), ("sample_count", "match_real_split"),
                          ("implementation", "ignite_default"), ("inception_input_size", 299)):
        if evaluation.get(key) != required:
            raise ValueError(f"evaluation.{key} must be {required!r}; got {evaluation.get(key)!r}.")
    return evaluation


def _extractor_metadata():
    weights = torchvision.models.Inception_V3_Weights.DEFAULT
    filename = weights.url.rsplit("/", 1)[-1]
    cached_file = Path(torch.hub.get_dir()) / "checkpoints" / filename
    return {
        "wrapper": "ignite.metrics.gan.utils.InceptionModel(return_features=False)",
        "weights_id": f"Inception_V3_Weights.{weights.name}",
        "weights_url": weights.url,
        "weights_file": str(cached_file) if cached_file.is_file() else None,
        "weights_sha256": _sha256(cached_file) if cached_file.is_file() else None,
        "output": "1000 post-softmax probabilities, no second softmax; not pool3 or logits",
    }


def main(argv=None):
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "outputs/resolution_128/best.pt")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--feature-device", choices=("cpu", "mps"))
    args = parser.parse_args(argv)
    checkpoint_path = _project_path(args.checkpoint)
    output_dir = _project_path(args.output_dir) if args.output_dir else checkpoint_path.parent / "evaluation"
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f"Output directory is not empty: {output_dir}; choose a new --output-dir.")

    # Inspect the checkpoint config safely before moving the model to any device.
    checkpoint_header = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = copy.deepcopy(checkpoint_header["config"])
    evaluation = _validate_config(config)
    seed = args.seed if args.seed is not None else evaluation.get("seed", 456)
    batch_size = args.batch_size if args.batch_size is not None else evaluation.get("batch_size", 32)
    device_name = args.feature_device if args.feature_device is not None else evaluation.get("device", "cpu")
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("Evaluation batch_size must be a positive integer.")
    if device_name not in ("cpu", "mps"):
        raise ValueError("Evaluation feature device must be cpu or mps.")
    device = torch.device(device_name)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; use --feature-device cpu.")
    config["training"]["batch_size"] = batch_size
    loaders = build_dataloaders(config)
    with tempfile.TemporaryDirectory() as temporary:
        manifest = save_split_manifest(loaders, config, Path(temporary) / "manifest.json")
    if manifest["fingerprint"] != checkpoint_header.get("split_fingerprint"):
        raise ValueError("Evaluation split fingerprint does not match the training checkpoint.")
    test_loader = loaders["test"]
    count = len(test_loader.dataset)
    if count < 2:
        raise ValueError("Evaluation requires at least two test images for FID.")
    paths = [entry["path"] for entry in manifest["splits"]["test"]]
    if len(paths) != count or test_loader.drop_last:
        raise ValueError("Test manifest/count mismatch or drop_last enabled.")
    checkpoint_hash = _sha256(checkpoint_path)
    print(f"Evaluating checkpoint epoch {checkpoint_header['epoch']}: test={count}, seed={seed}, "
          f"feature/generator device={device}, metric device=cpu", flush=True)
    model, _ = load_trained_model(checkpoint_path, device)
    print(f"Generating {count} fresh N(0,I) prior images...", flush=True)
    generated_images = generate_samples(model, count, seed, batch_size, device)
    del model
    if tuple(generated_images.shape) != (count, 1, config["data"]["image_size"], config["data"]["image_size"]):
        raise ValueError("Generated images must match the complete test count and native image geometry.")
    extractor = create_extractor(device)
    real = collect_probabilities(test_loader, extractor, device, count, FEATURE_DIMENSION, "real")
    generated_batches = (generated_images[start:start + batch_size] for start in range(0, count, batch_size))
    generated = collect_probabilities(generated_batches, extractor, device, count, FEATURE_DIMENSION, "generated")
    del extractor
    scores = compute_probability_metrics(real, generated, batch_size)
    metrics = {
        **scores, "num_real_samples": count, "num_generated_samples": count,
        "seed": seed, "batch_size": batch_size, "prior": "standard_normal_N(0,I)",
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": checkpoint_hash,
        "checkpoint_epoch": checkpoint_header["epoch"], "split_fingerprint": manifest["fingerprint"],
        "real_split": "test", "real_paths": paths, "test_class_counts": manifest["class_counts"]["test"],
        "feature_implementation": "ignite_default_1000_probabilities", "feature_dimension": FEATURE_DIMENSION,
        "extractor": _extractor_metadata(), "feature_device": str(device), "generator_device": str(device),
        "metric_device": "cpu", "metric_accumulation_dtype": "float64", "cached_probability_dtype": "float32",
        "preprocessing": {
            "native_image_size": config["data"]["image_size"], "input": "finite float grayscale [0,1]",
            "quantization": "ToPILImage float*255 cast uint8 (truncation), before resize",
            "resize": [299, 299], "interpolation": "PIL BILINEAR", "crop": None,
            "color": "RGB; three identical grayscale channels", "tensor": "ToTensor [0,1]",
            "normalize_mean": list(IMAGENET_MEAN), "normalize_std": list(IMAGENET_STD),
            "internal_transform_input": True, "range_rescaling": None,
            "tutorial_correction": "ImageNet input standardization supplied for pretrained wrapper transform_input=True",
        },
        "inception_score_definition": "exp(mean over ALL generated samples of KL(p(y|x)||mean p(y|x))); global, no splits or std",
        "versions": {"python": platform.python_version(), "torch": str(torch.__version__),
                     "torchvision": str(torchvision.__version__), "ignite": str(ignite.__version__),
                     "numpy": str(np.__version__), "scipy": str(scipy.__version__), "Pillow": str(PIL.__version__)},
        "elapsed_seconds": time.perf_counter() - started,
        "limitations": [
            "ImageNet classifier probabilities have a domain mismatch with chest radiographs; these scores do not establish clinical quality.",
            "Ignite default 1000-probability FID is nonstandard and cannot be compared numerically with 2048-pool3 FID.",
            f"Finite-sample FID/IS are biased/noisy; N={count}, D={FEATURE_DIMENSION}, covariance rank <= min(N-1,D). "
            "When N<D covariance is singular; SciPy sqrtm may warn or fail due to ill-conditioning.",
            "One prior seed and one final checkpoint; no uncertainty estimate, test-based selection, or clinical/quality winner claim.",
            manifest["grouping_note"],
        ],
    }
    # No artifacts are exported until membership, input/features, and both finite scores pass.
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    (output_dir / "evaluation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    torch.save({"real": real, "generated": generated, "real_paths": paths, "seed": seed,
                "checkpoint": str(checkpoint_path), "checkpoint_sha256": checkpoint_hash,
                "checkpoint_epoch": checkpoint_header["epoch"], "split_fingerprint": manifest["fingerprint"],
                "feature_implementation": metrics["feature_implementation"], "feature_dimension": FEATURE_DIMENSION,
                "preprocessing": metrics["preprocessing"], "extractor": metrics["extractor"]},
               output_dir / "feature_probabilities.pt")
    save_image(generated_images[:64], output_dir / "evaluation_generated_grid.png", nrow=8)
    print(f"FID={scores['fid']:.10g}; global IS={scores['inception_score']:.10g}; artifacts: {output_dir}", flush=True)
    return metrics


if __name__ == "__main__":
    main()

"""Evaluate saved diffusion samples using the PyTorch-Ignite GAN tutorial.

Reference: https://pytorch-ignite.ai/blog/gan-evaluation-with-fid-and-is/
The tutorial's default Ignite FID uses 1,000 class probabilities. We also
report FID on 2,048 pooled activations from the same torchvision network.
"""

import argparse
import json
from pathlib import Path

import ignite
import numpy as np
import torch
import torchvision
from ignite.metrics import FID, InceptionScore
from PIL import Image
from torchvision.models import Inception_V3_Weights, inception_v3


def resized_rgb(image):
    """Use the tutorial's PIL bilinear resize on actual RGB pixel values."""
    resized = image.convert("RGB").resize((299, 299), Image.Resampling.BILINEAR)
    return torch.from_numpy(np.asarray(resized).copy()).permute(2, 0, 1)


@torch.inference_mode()
def extract(items, model, classifier, device, batch_size):
    pooled, probabilities = [], []
    for start in range(0, len(items), batch_size):
        batch = []
        for item in items[start : start + batch_size]:
            if isinstance(item, Path):
                with Image.open(item) as image:
                    batch.append(resized_rgb(image))
            else:
                batch.append(resized_rgb(Image.fromarray(item)))
        pixels = torch.stack(batch).to(device=device, dtype=torch.float32).div_(255)
        features = model(pixels)
        pooled.append(features.cpu())
        probabilities.append(classifier(features).softmax(dim=1).cpu())
    return torch.cat(pooled), torch.cat(probabilities)


def frechet(first, second):
    # Identity extraction tells Ignite that it receives precomputed features.
    metric = FID(num_features=first.shape[1], feature_extractor=torch.nn.Identity())
    metric.update((first, second))
    return float(metric.compute())


def inception_score(probabilities):
    metric = InceptionScore(num_features=probabilities.shape[1], feature_extractor=torch.nn.Identity())
    metric.update(probabilities)
    return float(metric.compute())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--samples", type=Path, default=Path("results/samples"))
    parser.add_argument("--output", type=Path, default=Path("results/metrics.json"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=285, help="Seed for the real-training reference subset")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"],
                        default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("Batch size must be positive.")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    pixels = np.load(args.data / "images.npy", mmap_mode="r")
    with np.load(args.data / "split.npz") as split:
        train, test = split["train"], split["test"]
    generated = sorted(args.samples.glob("*.png"))
    n = len(test)
    if len(generated) != n or n < 2 or len(train) < n:
        parser.error(f"Expected {n} generated PNGs and at least {n} training examples. "
                     f"Found {len(generated)} PNGs and {len(train)} training examples.")
    train_control = np.random.default_rng(args.seed).choice(train, size=n, replace=False)

    weights = Inception_V3_Weights.IMAGENET1K_V1
    model = inception_v3(weights=weights).to(args.device).eval()
    classifier = model.fc
    model.fc = torch.nn.Identity()
    print(f"Extracting Inception features from {n} images in each of three sets", flush=True)
    test_features, test_probs = extract(pixels[test], model, classifier, args.device, args.batch_size)
    train_features, train_probs = extract(pixels[train_control], model, classifier, args.device, args.batch_size)
    fake_features, fake_probs = extract(generated, model, classifier, args.device, args.batch_size)
    print("Computing Fréchet distances and Inception Scores", flush=True)
    result = {
        "n_generated": n, "n_test": n, "n_train_reference": n,
        "fid_pool2048": frechet(fake_features, test_features),
        "fid_pool2048_train_vs_test": frechet(train_features, test_features),
        "fid_ignite_prob1000": frechet(fake_probs, test_probs),
        "fid_ignite_prob1000_train_vs_test": frechet(train_probs, test_probs),
        "is_generated": inception_score(fake_probs),
        "is_held_out": inception_score(test_probs),
        "is_train_reference": inception_score(train_probs),
        "weights": "torchvision Inception_V3_Weights.IMAGENET1K_V1",
        "preprocess": "PIL RGB bilinear 299x299, float [0,1], torchvision default internal input transform",
        "is_protocol": "one score over the entire sample, with no splits",
        "train_reference_seed": args.seed, "device": args.device, "batch_size": args.batch_size,
        "torch": str(torch.__version__), "torchvision": str(torchvision.__version__), "ignite": ignite.__version__,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

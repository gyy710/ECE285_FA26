"""Step 2: Load chest X-rays, preprocess them, inspect a batch, and save a preview.

Run: python dataset.py
Interface: build_dataloaders(config) returns train / val / test DataLoaders.
Each batch is (images, labels), with float32 images of shape [B,1,H,W] in [0,1].
When validation_fraction>0, hold out validation data from the original train
split by class and candidate filename group. The original val split is unused;
test always uses its original separate directory.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.datasets.folder import IMG_EXTENSIONS
from torchvision.utils import save_image


PROJECT_ROOT = Path(__file__).resolve().parent
SPLITS = ("train", "val", "test")
CLASSES = ("NORMAL", "PNEUMONIA")


def _project_path(value):
    """Resolve relative configuration paths against the project directory."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _is_image(path):
    # Exclude macOS metadata such as ._xxx.jpeg before image decoding.
    return not Path(path).name.startswith(".") and path.lower().endswith(IMG_EXTENSIONS)


def filename_group_key(path, class_name=None):
    """Extract a conservative candidate filename group, not a verified patient ID.

    NORMAL keeps the IM / NORMAL2-IM namespaces separate; PNEUMONIA groups
    bacteria/virus filenames with the same personN. Numbers may be reused,
    so grouping only reduces the risk of potential leakage.
    """
    path = Path(path)
    label = class_name or path.parent.name
    if label == "NORMAL":
        match = re.match(r"^(?:NORMAL2-IM|IM)-\d+", path.stem)
        if match:
            return f"{label}:{match.group(0)}"
    elif label == "PNEUMONIA":
        match = re.match(r"^person(\d+)_", path.stem)
        if match:
            return f"{label}:person{int(match.group(1))}"
    return f"{label}:{path.stem}"


class ImageSubset(Subset):
    """Preserve ImageFolder statistics while delegating reads to the source Dataset."""
    def __init__(self, dataset, indices):
        super().__init__(dataset, sorted(indices))
        self.samples = [dataset.samples[index] for index in self.indices]
        self.targets = [target for _, target in self.samples]
        self.classes = dataset.classes
        self.class_to_idx = dataset.class_to_idx
        self.root = dataset.root


def _hold_out_validation(dataset, fraction, seed):
    """Hold out about fraction of each class's groups without splitting a group."""
    groups_by_class = defaultdict(lambda: defaultdict(list))
    for index, (path, target) in enumerate(dataset.samples):
        group = filename_group_key(path, dataset.classes[target])
        groups_by_class[target][group].append(index)
    rng = random.Random(seed)
    train_indices, val_indices = [], []
    for target in sorted(groups_by_class):
        groups = groups_by_class[target]
        keys = sorted(groups)
        if len(keys) < 2:
            raise ValueError(f"Class {dataset.classes[target]} needs at least two candidate groups for a train/validation split.")
        rng.shuffle(keys)
        count = max(1, min(len(keys) - 1, round(len(keys) * fraction)))
        for position, key in enumerate(keys):
            destination = val_indices if position < count else train_indices
            destination.extend(groups[key])
    return ImageSubset(dataset, train_indices), ImageSubset(dataset, val_indices)


def build_dataloaders(config):
    """Shuffle only train; retain final batches and exclude test from the holdout.

    ImageFolder provides labels for statistics; VAE training uses only images.
    Avoid scanning every directory under root, which could include nested copies.
    """
    data_config = config["data"]
    root = _project_path(data_config["root"])
    fraction = data_config.get("validation_fraction", 0.0)
    if not 0 <= fraction < 1:
        raise ValueError("data.validation_fraction must be in [0,1); 0 keeps the original splits.")
    if data_config["channels"] != 1:
        raise ValueError("This chest X-ray VAE uses single-channel images; set data.channels to 1.")
    for split in SPLITS:
        for label in CLASSES:
            folder = root / split / label
            if not folder.is_dir():
                raise FileNotFoundError(
                    f"Missing data directory: {folder}. data.root must contain train/val/test directories."
                )

    size = data_config["image_size"]
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((size, size), antialias=True),
        transforms.ToTensor(),  # uint8 pixels [0,255] -> float32 [0,1]
    ])

    split_datasets = {}
    for split in SPLITS:
        dataset = datasets.ImageFolder(
            root=root / split,
            transform=transform,
            is_valid_file=_is_image,
        )
        if dataset.classes != list(CLASSES):
            raise ValueError(f"Classes in {root / split} must be exactly {CLASSES}, got {dataset.classes}.")
        split_datasets[split] = dataset
    if fraction > 0:
        split_datasets["train"], split_datasets["val"] = _hold_out_validation(
            split_datasets["train"], fraction, config["seed"]
        )

    loaders = {}
    for split, dataset in split_datasets.items():
        loaders[split] = DataLoader(
            dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=(split == "train"),
            num_workers=data_config["num_workers"],
            drop_last=False,
            generator=torch.Generator().manual_seed(config["seed"]),
        )
    return loaders


def save_split_manifest(loaders, config, output_path):
    """Save file membership and a fingerprint for reproducibility and resume checks."""
    root = _project_path(config["data"]["root"])
    fraction = config["data"].get("validation_fraction", 0.0)
    split_entries, class_counts, groups = {}, {}, {}
    for split, loader in loaders.items():
        dataset = loader.dataset
        split_entries[split] = [
            {
                "path": str(Path(path).relative_to(root)),
                "label": dataset.classes[target],
                "filename_group": filename_group_key(path, dataset.classes[target]),
            }
            for path, target in dataset.samples
        ]
        counts = Counter(dataset.targets)
        class_counts[split] = {
            name: counts[index] for name, index in dataset.class_to_idx.items()
        }
        groups[split] = {entry["filename_group"] for entry in split_entries[split]}
    protocol = "class_stratified_filename_groups_from_train" if fraction else "original_splits"
    identity = {"protocol": protocol, "seed": config["seed"], "splits": split_entries}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    unused_val = []
    if fraction:
        original_val = datasets.ImageFolder(root=root / "val", is_valid_file=_is_image)
        unused_val = [str(Path(path).relative_to(root)) for path, _ in original_val.samples]
    manifest = {
        **identity,
        "fingerprint": fingerprint,
        "validation_fraction_requested": fraction,
        "counts": {split: len(loader.dataset) for split, loader in loaders.items()},
        "class_counts": class_counts,
        "unused_original_val": unused_val,
        "filename_group_overlap": {
            "train_val": len(groups["train"] & groups["val"]),
            "train_test": len(groups["train"] & groups["test"]),
            "val_test": len(groups["val"] & groups["test"]),
        },
        "grouping_note": (
            "Filename groups are unverified identity proxies; matching person numbers across "
            "bacteria/virus may refer to different people. Group separation reduces possible "
            "train/validation leakage but does not establish patient independence. Original "
            "test remains unchanged and its ambiguous filename overlaps are recorded."
        ),
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    loaders = build_dataloaders(config)

    for split, loader in loaders.items():
        counts = Counter(loader.dataset.targets)
        details = ", ".join(
            f"{name}={counts[index]}"
            for name, index in loader.dataset.class_to_idx.items()
        )
        print(f"{split}: {len(loader.dataset)} images ({details})")

    images, _ = next(iter(loaders["train"]))
    print(f"Batch shape: {tuple(images.shape)}")
    print(f"Batch dtype: {images.dtype}; pixel range: [{images.min():.3f}, {images.max():.3f}]")
    output_dir = _project_path(config["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    preview = output_dir / "real_samples.png"
    save_image(images[:64], preview, nrow=8, padding=2)
    print(f"Preview saved: {preview}")
    manifest_path = output_dir / "split_manifest.json"
    manifest = save_split_manifest(loaders, config, manifest_path)
    print(f"Split protocol: {manifest['protocol']}; manifest: {manifest_path}")
    if manifest["unused_original_val"]:
        print(f"Original val images reserved and unused: {len(manifest['unused_original_val'])}")


if __name__ == "__main__":
    main()

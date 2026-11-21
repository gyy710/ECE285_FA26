"""Prepare the Histopathology subset of Kaggle's Scientific Image Classification Dataset."""
import argparse
import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile
import re
import shutil

import numpy as np
from PIL import Image, ImageOps

URL = "https://www.kaggle.com/api/v1/datasets/download/rushilprajapati/data-final?datasetVersionNumber=1"
PREFIX = "data/Histopathology/"
SEED = 285


def split_groups(groups, hashes):
    """Keep numeric filename families and any exact model-input duplicates together."""
    parent = list(range(len(groups)))

    def root(i):
        while i != parent[i]:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for keys in (groups, hashes):
        seen = {}
        for i, key in enumerate(keys):
            parent[root(i)] = root(seen.setdefault(key, i))

    members = defaultdict(list)
    for i in range(len(groups)):
        members[root(i)].append(i)
    families = list(members.values())
    test = []
    for idx in np.random.default_rng(SEED).permutation(len(families)):
        test.extend(families[idx])
        if len(test) >= round(len(groups) / 10):
            break
    test = np.array(sorted(test), dtype=np.int64)
    train = np.setdiff1d(np.arange(len(groups)), test)
    return train, test, len(families)


def prepare(archive, out):
    out.mkdir(parents=True, exist_ok=True)
    rows, hashes, groups = [], [], []
    sizes = Counter()
    with ZipFile(archive) as source:
        paths = sorted(n for n in source.namelist() if n.startswith(PREFIX) and not n.endswith("/"))
        if len(paths) != 5179:
            raise ValueError(f"Expected 5179 files in {PREFIX}, found {len(paths)}")
        images = np.empty((len(paths), 64, 64, 3), dtype=np.uint8)
        for i, path in enumerate(paths):
            with Image.open(io.BytesIO(source.read(path))) as im:
                sizes[im.size] += 1
                image = ImageOps.fit(im.convert("RGB"), (64, 64), method=Image.Resampling.LANCZOS)
                images[i] = np.asarray(image)
            hashes.append(hashlib.sha256(images[i].tobytes()).hexdigest())
            name = Path(path).name
            groups.append(re.fullmatch(r"image_(\d+)(?: \(\d+\))?\.png", name)[1])
            rows.append([i, path, groups[-1], hashes[-1]])
    train, test, family_count = split_groups(groups, hashes)
    test_ids = set(test.tolist())
    with (out / "manifest.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "archive_path", "numeric_stem", "input_sha256", "split"])
        for i, row in enumerate(rows):
            writer.writerow(row + ["test" if i in test_ids else "train"])
    np.save(out / "images.npy", images)
    np.savez(out / "split.npz", train=train, test=test)
    metadata = dict(dataset="rushilprajapati/data-final", version=1, folder=PREFIX, files=len(rows),
                    train=len(train), test=len(test), split_seed=SEED, input_shape=list(images.shape),
                    preprocessing="RGB, centered square crop and Lanczos resampling to 64x64",
                    native_sizes={f"{w}x{h}": n for (w,h),n in sorted(sizes.items())},
                    split_groups=family_count, exact_duplicate_excess=len(rows)-len(set(hashes)),
                    unique_numeric_stems=len(set(groups)), url=URL)
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("data/data-final-v1.zip"))
    parser.add_argument("--out", type=Path, default=Path("data"))
    args = parser.parse_args()
    if not args.archive.exists():
        args.archive.parent.mkdir(parents=True, exist_ok=True)
        partial = args.archive.with_suffix(".zip.part")
        with urlopen(URL, timeout=120) as source, partial.open("wb") as target:
            shutil.copyfileobj(source, target, 1024**2)
        partial.replace(args.archive)
    prepare(args.archive, args.out)

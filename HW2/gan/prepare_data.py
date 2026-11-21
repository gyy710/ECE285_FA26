"""Convert the Kaggle WikiArt archive (steubk/wikiart) into a 64x64 uint8 array.

Images are read straight from archive.zip (no 30+ GB extraction), resized so the
shorter side is 64 px (antialiased bicubic), and center-cropped to 64x64.
"""

import argparse
import csv
import io
import zipfile
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None
IMG_EXT = (".jpg", ".jpeg", ".png")

_zip = None
_size = None


def _init(zip_path, size):
    global _zip, _size
    _zip = zipfile.ZipFile(zip_path)
    _size = size


def _load(name):
    try:
        img = Image.open(io.BytesIO(_zip.read(name)))
        img.draft("RGB", (_size * 2, _size * 2))  # fast DCT downscaling for JPEGs
        img = img.convert("RGB")
        w, h = img.size
        s = _size / min(w, h)
        nw, nh = max(_size, round(w * s)), max(_size, round(h * s))
        img = img.resize((nw, nh), Image.BICUBIC, reducing_gap=3.0)
        left, top = (nw - _size) // 2, (nh - _size) // 2
        img = img.crop((left, top, left + _size, top + _size))
        return name, np.asarray(img, dtype=np.uint8)
    except Exception as e:  # corrupted / unreadable file
        return name, repr(e)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zip", default="data/archive.zip")
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--out", default="data/wikiart64")
    p.add_argument("--workers", type=int, default=12)
    args = p.parse_args()

    with zipfile.ZipFile(args.zip) as zf:
        names = sorted(n for n in zf.namelist() if n.lower().endswith(IMG_EXT) and "/" in n)
    print(f"{len(names)} images in archive")

    images, kept, failed = [], [], []
    with Pool(args.workers, initializer=_init, initargs=(args.zip, args.size)) as pool:
        for name, res in tqdm(pool.imap(_load, names, chunksize=64), total=len(names)):
            if isinstance(res, str):
                failed.append((name, res))
            else:
                images.append(res)
                kept.append(name)

    arr = np.stack(images)
    np.save(args.out + ".npy", arr)
    with open(args.out + "_meta.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "file", "style"])
        for i, n in enumerate(kept):
            w.writerow([i, n, n.split("/")[-2]])
    styles = Counter(n.split("/")[-2] for n in kept)
    print(f"saved {arr.shape} to {args.out}.npy; failed={len(failed)}")
    for n, e in failed[:10]:
        print("  failed:", n, e)
    print(f"{len(styles)} styles:")
    for s, c in styles.most_common():
        print(f"  {s:35s} {c}")


if __name__ == "__main__":
    main()

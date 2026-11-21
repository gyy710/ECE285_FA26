"""Rebuild the sample comparison and loss figure from the completed experiment."""
from pathlib import Path
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "report" / "figures"


def grid(images, columns=5):
    size = 64
    gap = 3
    rows = (len(images) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * (size + gap) - gap, rows * (size + gap) - gap), "white")
    for index, image in enumerate(images):
        sheet.paste(Image.fromarray(image), ((index % columns) * (size + gap), (index // columns) * (size + gap)))
    return sheet


def sample_comparison():
    pixels = np.load(ROOT / "data" / "images.npy", mmap_mode="r")
    with np.load(ROOT / "data" / "split.npz") as split:
        selected = np.random.default_rng(19).choice(split["test"], 15, replace=False)
    grid(pixels[selected]).save(FIGURES / "held_out.png")
    images = []
    for path in sorted((ROOT / "results" / "samples").glob("*.png"))[:15]:
        with Image.open(path) as source:
            images.append(np.asarray(source.convert("RGB")))
    if len(images) != 15:
        raise ValueError("Expected at least 15 generated images")
    grid(images).save(FIGURES / "generated.png")


def training_curve():
    with (ROOT / "results" / "training.csv").open() as log:
        records = list(csv.DictReader(log))
    step = np.array([int(row["step"]) for row in records])
    mse = np.array([float(row["mse"]) for row in records])
    # Each band is the range across one group of 100 updates.
    starts = np.arange(0, len(step), 100)
    if len(step) % 100:
        raise ValueError("The final experiment should end on a complete 100-step group")
    bins = mse.reshape(-1, 100)
    x = (step[starts] + step[starts + 99]) / 2
    fig, ax = plt.subplots(figsize=(2.88, 1.53), constrained_layout=True)
    ax.fill_between(x, bins.min(1), bins.max(1), color="#dbcfe3", label="Min–max within 100 steps")
    ax.plot(x, bins.mean(1), color="#512567", lw=1.7, label="100-step mean")
    ax.set(xlabel="Optimizer update (thousands)", ylabel="Velocity MSE", yscale="log")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, pos: f"{value / 1000:g}"))
    ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.1, 0.2, 0.4, 0.8]))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.margins(x=.012)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#dedede", linewidth=.5)
    ax.tick_params(labelsize=8)
    fig.savefig(FIGURES / "loss.pdf")
    plt.close(fig)
    return len(step), bins.mean(1)[0], bins.mean(1)[-1]


def save_measurements(steps, first_loss, last_loss):
    scores = json.loads((ROOT / "results" / "metrics.json").read_text())
    values = {
        "updateCount": f"{steps:,}",
        "firstLoss": f"{first_loss:.3f}",
        "lastLoss": f"{last_loss:.3f}",
        "genIS": f"{scores['is_generated']:.2f}",
        "genFID": f"{scores['fid_pool2048']:.2f}",
        "probFID": f"{scores['fid_ignite_prob1000']:.4f}",
        "refFID": f"{scores['fid_pool2048_train_vs_test']:.2f}",
        "refProbFID": f"{scores['fid_ignite_prob1000_train_vs_test']:.4f}",
        "trainIS": f"{scores['is_train_reference']:.2f}",
    }
    tex = "".join(f"\\newcommand{{\\{name}}}{{{value}}}\n" for name, value in values.items())
    (ROOT / "report" / "measurements.tex").write_text(tex)


if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    sample_comparison()
    save_measurements(*training_curve())

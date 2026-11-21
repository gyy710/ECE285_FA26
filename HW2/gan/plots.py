"""Figures for the report: loss curves, IS/FID curves, sample grids, training progression."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
import torch
import torchvision.utils as vutils
from PIL import Image

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "legend.fontsize": 7, "font.family": "serif"})


def smooth(y, k=200):
    return pd.Series(y).rolling(k, min_periods=1).mean().to_numpy()


def loss_figure(runs, labels, out, extra_metrics=None):
    fig, axes = plt.subplots(1, len(runs) + 2, figsize=(7.0, 1.75))
    for ax, run, label in zip(axes, runs, labels):
        df = pd.read_csv(Path(run) / "losses.csv")
        it = df["iteration"].to_numpy()
        for col, c in [("Loss_D", "tab:blue"), ("Loss_G", "tab:red")]:
            ax.plot(it, df[col], color=c, alpha=0.15, lw=0.3)
            ax.plot(it, smooth(df[col]), color=c, lw=1.0, label=col.replace("Loss_", "") + " loss")
        ax.set_title(f"Losses, {label}")
        ax.set_xlabel("iteration")
        ax.set_ylim(0, 8)
        ax.legend(loc="upper right", frameon=False)
        ax.grid(alpha=0.3)
    curves = [(pd.read_csv(Path(r) / "metrics.csv"), l) for r, l in zip(runs, labels)]
    if extra_metrics:
        curves.append((pd.read_csv(extra_metrics[0]), extra_metrics[1]))
    ax_fid, ax_is = axes[-2], axes[-1]
    for (m, label), c in zip(curves, ["tab:blue", "tab:orange", "tab:gray"]):
        ax_fid.plot(m["epoch"], m["FID2048"], color=c, lw=1.0, label=label.split(":")[0])
        ax_is.plot(m["epoch"], m["IS"], color=c, lw=1.0, label=label.split(":")[0])
    ax_fid.set_yscale("log")
    ax_fid.set_yticks([30, 50, 100, 200, 300], ["30", "50", "100", "200", "300"])
    ax_fid.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax_fid.set_title("FID (pool3) $\\downarrow$, 2k samples")
    ax_is.set_title("IS $\\uparrow$, 2k samples")
    for ax in (ax_fid, ax_is):
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, loc="upper right" if ax is ax_fid else "lower right")
    fig.tight_layout(pad=0.3, w_pad=0.6)
    fig.savefig(out, dpi=300)
    plt.close(fig)


def grid_from_tensor(x, n, nrow):
    return vutils.make_grid(x[:n], nrow=nrow, padding=1, normalize=True, value_range=(-1, 1), pad_value=1.0)


def progression_tensor(run, epochs, k):
    """First k fixed-noise samples at each epoch, from the saved 8x8 grids (padding 2)."""
    tiles = []
    for ep in epochs:
        img = np.asarray(Image.open(Path(run) / "samples" / f"epoch_{ep:03d}.png").convert("RGB"))
        for i in range(k):
            r, c = i // 8, i % 8
            tiles.append(img[2 + r * 66 : 2 + r * 66 + 64, 2 + c * 66 : 2 + c * 66 + 64])
    return torch.from_numpy(np.stack(tiles)).permute(0, 3, 1, 2).float() / 127.5 - 1


def samples_figure(data_path, sample_files, labels, out, n=36, nrow=6, progress=None):
    d = np.load(data_path, mmap_mode="r")
    idx = np.sort(np.random.default_rng(1).choice(len(d), n, replace=False))
    real = torch.from_numpy(np.ascontiguousarray(d[idx])).permute(0, 3, 1, 2).float() / 127.5 - 1
    grids = [grid_from_tensor(real, n, nrow)] + [grid_from_tensor(torch.load(f), n, nrow) for f in sample_files]
    titles = ["Real WikiArt (64$\\times$64)"] + labels
    if progress is not None:
        run, epochs = progress
        grids.append(grid_from_tensor(progression_tensor(run, epochs, nrow), nrow * len(epochs), nrow))
        titles.append(f"{labels[0].split(':')[0]}: epochs " + ", ".join(map(str, epochs)))
    fig, axes = plt.subplots(1, len(grids), figsize=(7.0, 7.0 / len(grids) + 0.2))
    for ax, g, t in zip(axes, grids, titles):
        ax.imshow(g.permute(1, 2, 0).numpy())
        ax.set_title(t, pad=2, fontsize=7)
        ax.axis("off")
    fig.subplots_adjust(left=0.003, right=0.997, bottom=0.003, top=0.91, wspace=0.03)
    fig.savefig(out, dpi=300)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", default=["runs/dcgan_bce", "runs/dcgan_bce_ema"])
    p.add_argument("--labels", nargs="+", default=["A: DCGAN", "B: DCGAN+EMA"])
    p.add_argument("--samples", nargs="+",
                   default=["runs/dcgan_bce/final_samples_last_G.pt", "runs/dcgan_bce_ema/final_samples_last_G_ema.pt"])
    p.add_argument("--extra_metrics", nargs=2, default=["results/sngan_hinge_ema_aborted_metrics.csv", "SN+hinge"])
    p.add_argument("--data", default="data/wikiart64.npy")
    p.add_argument("--out", default="report/figures")
    p.add_argument("--progress_run", default="runs/dcgan_bce")
    p.add_argument("--progress_epochs", nargs="+", type=int, default=[1, 2, 5, 10, 20, 40])
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    loss_figure(args.runs, args.labels, out / "curves.pdf", args.extra_metrics)
    samples_figure(args.data, args.samples, args.labels, out / "samples.pdf",
                   progress=(args.progress_run, args.progress_epochs))
    print("figures written to", out)


if __name__ == "__main__":
    main()

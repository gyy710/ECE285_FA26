"""Generate independent PNG samples using the EMA network and DDIM."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from diffusion import Denoiser, Diffusion
from train import default_device


@torch.inference_mode()
def run(args):
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    cfg = checkpoint["config"]
    model = Denoiser(cfg["width"])
    model.load_state_dict(checkpoint["model"])
    model.to(args.device, memory_format=torch.channels_last).eval()
    schedule = Diffusion(cfg["timesteps"]).to(args.device)
    count = args.count if args.count is not None else len(np.load(args.split)["test"])
    out = Path(args.out)
    out.mkdir(exist_ok=True, parents=True)
    if any(out.glob("*.png")):
        raise FileExistsError(f"Select an empty --out directory to avoid mixing sample runs: {out}")
    rng = torch.Generator(device="cpu").manual_seed(args.seed)
    for start in range(0, count, args.batch_size):
        n = min(args.batch_size, count - start)
        shape = (n, 3, cfg["image_size"], cfg["image_size"])
        latent = torch.randn(shape, generator=rng).to(args.device, memory_format=torch.channels_last)
        sampled = schedule.sample(model, latent, args.steps)
        pixels = ((sampled + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()
        for offset, pixel_array in enumerate(pixels):
            Image.fromarray(pixel_array).save(out / f"{start + offset:05d}.png")
        print(f"generated {start + n}/{count}", flush=True)
    (out / "sampling.json").write_text(json.dumps(dict(vars(args), count=count, training_step=checkpoint["step"]), indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="results/checkpoint.pt")
    parser.add_argument("--split", default="data/split.npz")
    parser.add_argument("--out", default="results/samples")
    parser.add_argument("--count", type=int)
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=286)
    parser.add_argument("--device", default=default_device())
    run(parser.parse_args())

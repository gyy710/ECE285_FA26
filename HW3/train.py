"""Fit the unconditional diffusion model on the prepared training split."""
import argparse
import copy
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from diffusion import Denoiser, Diffusion


def default_device():
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def run(args):
    out = Path(args.out)
    out.mkdir(exist_ok=True, parents=True)
    if (out / "training.csv").exists():
        raise FileExistsError(f"Training output already exists in {out}. Choose a new --out.")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    source = np.load(Path(args.data) / "images.npy", mmap_mode="r")
    train_indices = np.load(Path(args.data) / "split.npz")["train"]
    images = torch.from_numpy(source[train_indices]).to(args.device)
    model = Denoiser(args.width).to(args.device, memory_format=torch.channels_last)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    schedule = Diffusion(args.timesteps).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    config = dict(vars(args), train_count=len(images), parameters=sum(p.numel() for p in model.parameters()),
                  torch_version=str(torch.__version__), prediction="velocity", image_size=source.shape[1])
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps(config), flush=True)
    start = time.monotonic()
    with (out / "training.csv").open("w") as log:
        writer = csv.writer(log)
        writer.writerow(["step", "epoch", "images_seen", "mse", "lr", "elapsed_seconds"])
        seen = step = epoch = 0
        while step < args.updates:
            epoch += 1
            order = rng.permutation(len(images))
            for begin in range(0, len(order), args.batch_size):
                step += 1
                selected = torch.as_tensor(order[begin:begin + args.batch_size], device=args.device)
                clean = images[selected].permute(0, 3, 1, 2).float().div_(127.5).sub_(1)
                if rng.random() < .5:
                    clean = clean.flip([3])
                if rng.random() < .5:
                    clean = clean.flip([2])
                # Linear warmup followed by a cosine taper to one fifth the peak.
                warmup = min(100, args.updates // 10)
                progress = max(step - warmup, 0) / max(args.updates - warmup, 1)
                lr = args.lr * min(1., step / max(1, warmup)) * (.6 + .4 * math.cos(math.pi * progress))
                optimizer.param_groups[0]["lr"] = lr
                optimizer.zero_grad(set_to_none=True)
                loss = schedule.loss(model, clean)
                loss.backward()
                optimizer.step()
                # Amortize EMA updates over five optimizer steps.
                if step % 5 == 0 or step == args.updates:
                    decay = min(args.ema, (step - 1) / (step + 9))
                    with torch.no_grad():
                        torch._foreach_lerp_(list(ema.parameters()), list(model.parameters()), 1 - decay ** 5)
                value = loss.item()
                if not math.isfinite(value):
                    raise FloatingPointError(f"Non-finite training loss at step {step}")
                seen += len(clean)
                writer.writerow([step, epoch, seen, value, lr, round(time.monotonic() - start, 3)])
                if step == 1 or step % 50 == 0 or step == args.updates:
                    log.flush()
                    print(f"step {step} / {args.updates}, mse {value:.4f}, elapsed {time.monotonic() - start:.1f}s", flush=True)
                if step % 500 == 0 or step == args.updates:
                    path = out / "checkpoint.pt"
                    torch.save({"model": ema.state_dict(), "config": config, "step": step}, path.with_suffix(".tmp"))
                    path.with_suffix(".tmp").replace(path)
                if step == args.updates:
                    break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default="results")
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--timesteps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--updates", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=.0006)
    parser.add_argument("--ema", type=float, default=.995)
    parser.add_argument("--seed", type=int, default=285)
    run(parser.parse_args())

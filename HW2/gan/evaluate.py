"""Final IS / FID evaluation of trained generators (ignite metrics, see metrics.py)."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torchvision.utils as vutils
import ignite

from metrics import GANEvaluator
from models import Generator, recalibrate_bn


def load_generator(ckpt_path, device, use_ema=True):
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = state["args"]
    G = Generator(a["latent_dim"], a["ngf"])
    key = "G_ema" if use_ema and "G_ema" in state else "G"
    G.load_state_dict(state[key])
    G = G.to(device).eval()
    if key == "G_ema":
        recalibrate_bn(G, a["latent_dim"], device)
    return G, a, state["epoch"], key


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True, help="run directories")
    p.add_argument("--ckpt", default="last.pt")
    p.add_argument("--data", default="data/wikiart64.npy")
    p.add_argument("--n", type=int, default=10000)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--out", default="results/final_metrics.json")
    p.add_argument("--raw", action="store_true", help="evaluate online G weights even if EMA exists")
    args = p.parse_args()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    data = np.load(args.data, mmap_mode="r")
    idx = np.sort(np.random.default_rng(args.seed).choice(len(data), args.n, replace=False))
    real = torch.from_numpy(np.ascontiguousarray(data[idx])).permute(0, 3, 1, 2).contiguous()

    evaluator = GANEvaluator(device)
    results = {}
    for i, run in enumerate(args.runs):
        run = Path(run)
        ignite.utils.manual_seed(args.seed)
        G, a, epoch, key = load_generator(run / "ckpt" / args.ckpt, device, use_ema=not args.raw)
        m = evaluator.evaluate(G, real, a["latent_dim"], real_is=(i == 0))
        m.update(epoch=epoch, weights=key, n=args.n)
        results[run.name] = m
        print(run.name, m)

        z = torch.randn(64, a["latent_dim"], 1, 1, device=device, generator=torch.Generator(device).manual_seed(0))
        with torch.no_grad():
            fake = G(z).cpu()
        tag = f"{args.ckpt[:-3]}_{key}"
        vutils.save_image(fake, run / f"final_samples_{tag}.png", nrow=8, padding=2,
                          normalize=True, value_range=(-1, 1))
        torch.save(fake, run / f"final_samples_{tag}.pt")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()

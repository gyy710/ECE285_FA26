import argparse
import copy
import csv
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as vutils
import ignite
from ignite.engine import Engine, Events

from models import Discriminator, Generator, count_params, initialize_fn, recalibrate_bn
from metrics import GANEvaluator


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/wikiart64.npy")
    p.add_argument("--out", default="runs/dcgan")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--latent_dim", type=int, default=100)
    p.add_argument("--ngf", type=int, default=64)
    p.add_argument("--ndf", type=int, default=64)
    p.add_argument("--lr_g", type=float, default=2e-4)
    p.add_argument("--lr_d", type=float, default=2e-4)
    p.add_argument("--beta1", type=float, default=0.5)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--loss", choices=["bce", "hinge"], default="bce")
    p.add_argument("--spectral", action="store_true", help="spectral norm in D (no BN)")
    p.add_argument("--ema", type=float, default=0.0, help="EMA decay for G weights; 0 disables")
    p.add_argument("--hflip", action="store_true", help="random horizontal flip augmentation")
    p.add_argument("--eval_every", type=int, default=1, help="epochs between IS/FID evaluations")
    p.add_argument("--eval_n", type=int, default=2000, help="#real/#fake images for per-epoch IS/FID")
    p.add_argument("--max_iters", type=int, default=0, help="debug: stop after this many iterations")
    p.add_argument("--seed", type=int, default=999)
    return p.parse_args()


class GPUBatches:
    """Iterates over random mini-batches of a uint8 image tensor kept on the device."""

    def __init__(self, images, batch_size, hflip, device):
        self.images = images
        self.batch_size = batch_size
        self.hflip = hflip
        self.device = device

    def __len__(self):
        return self.images.shape[0] // self.batch_size

    def __iter__(self):
        perm = torch.randperm(self.images.shape[0], device=self.device)
        for i in range(len(self)):
            x = self.images[perm[i * self.batch_size : (i + 1) * self.batch_size]].float() / 127.5 - 1
            if self.hflip:
                flip = torch.rand(x.shape[0], device=self.device) < 0.5
                x = torch.where(flip[:, None, None, None], x.flip(3), x)
            yield x


def main():
    args = get_args()
    out = Path(args.out)
    (out / "samples").mkdir(parents=True, exist_ok=True)
    (out / "ckpt").mkdir(exist_ok=True)
    json.dump(vars(args), open(out / "config.json", "w"), indent=2)

    ignite.utils.manual_seed(args.seed)
    logger = ignite.utils.setup_logger("train", filepath=str(out / "train.log"))
    logging.getLogger("ignite.engine.engine.Engine").setLevel(logging.WARNING)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    data = torch.from_numpy(np.load(args.data))  # (N, 64, 64, 3) uint8
    data = data.permute(0, 3, 1, 2).contiguous()
    rng = np.random.default_rng(0)
    eval_real = data[torch.from_numpy(rng.choice(len(data), args.eval_n, replace=False))]
    train_data = GPUBatches(data.to(device), args.batch_size, args.hflip, device)
    logger.info(f"device={device} images={len(data)} iters/epoch={len(train_data)}")

    netG = Generator(args.latent_dim, args.ngf).to(device)
    netD = Discriminator(args.ndf, spectral=args.spectral).to(device)
    netG.apply(initialize_fn)
    netD.apply(initialize_fn)
    netG_ema = copy.deepcopy(netG).eval() if args.ema > 0 else None
    logger.info(f"G params={count_params(netG):,} D params={count_params(netD):,}")

    optimizerD = optim.Adam(netD.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))
    fixed_noise = torch.randn(64, args.latent_dim, 1, 1, device=device)

    def d_loss_fn(real_logits, fake_logits):
        if args.loss == "hinge":
            return F.relu(1 - real_logits).mean() + F.relu(1 + fake_logits).mean()
        return (
            F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
            + F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
        )

    def g_loss_fn(fake_logits):
        if args.loss == "hinge":
            return -fake_logits.mean()
        return F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))

    @torch.no_grad()
    def update_ema():
        for pe, p in zip(netG_ema.parameters(), netG.parameters()):
            pe.lerp_(p, 1 - args.ema)
        for be, b in zip(netG_ema.buffers(), netG.buffers()):
            be.copy_(b)

    def training_step(engine, real):
        netG.train()
        netD.train()
        b_size = real.size(0)

        # (1) Update D: maximize log D(x) + log(1 - D(G(z)))  (or hinge)
        optimizerD.zero_grad(set_to_none=True)
        noise = torch.randn(b_size, args.latent_dim, 1, 1, device=device)
        fake = netG(noise)
        out_real = netD(real)
        out_fake = netD(fake.detach())
        errD = d_loss_fn(out_real, out_fake)
        errD.backward()
        optimizerD.step()

        # (2) Update G: maximize log D(G(z))  (non-saturating)  (or -D(G(z)) for hinge)
        optimizerG.zero_grad(set_to_none=True)
        out_fake2 = netD(fake)
        errG = g_loss_fn(out_fake2)
        errG.backward()
        optimizerG.step()
        if netG_ema is not None:
            update_ema()

        return {
            "Loss_G": errG.item(),
            "Loss_D": errD.item(),
            "D_x": torch.sigmoid(out_real).mean().item(),
            "D_G_z1": torch.sigmoid(out_fake).mean().item(),
            "D_G_z2": torch.sigmoid(out_fake2).mean().item(),
        }

    trainer = Engine(training_step)
    evaluator = GANEvaluator(device)
    sampler = lambda: netG_ema if netG_ema is not None else netG

    loss_file = open(out / "losses.csv", "w", newline="")
    loss_writer = csv.writer(loss_file)
    loss_writer.writerow(["iteration", "epoch", "Loss_G", "Loss_D", "D_x", "D_G_z1", "D_G_z2"])
    metric_file = open(out / "metrics.csv", "w", newline="")
    metric_writer = csv.writer(metric_file)
    metric_writer.writerow(["epoch", "iteration", "IS", "FID", "FID2048", "train_time_s"])
    t_start = time.time()
    best = {"fid2048": float("inf")}

    @trainer.on(Events.ITERATION_COMPLETED)
    def store_losses(engine):
        o = engine.state.output
        loss_writer.writerow([engine.state.iteration, engine.state.epoch] + [f"{o[k]:.5f}" for k in o])
        if engine.state.iteration % 100 == 0:
            loss_file.flush()
            logger.info(
                f"ep {engine.state.epoch} it {engine.state.iteration} "
                + " ".join(f"{k}={v:.3f}" for k, v in o.items())
            )
        if args.max_iters and engine.state.iteration >= args.max_iters:
            engine.terminate()

    @trainer.on(Events.EPOCH_COMPLETED)
    def end_of_epoch(engine):
        ep = engine.state.epoch
        G = sampler()
        if netG_ema is not None:
            recalibrate_bn(netG_ema, args.latent_dim, device)
        G.eval()
        with torch.no_grad():
            fake = G(fixed_noise).cpu()
        vutils.save_image(fake, out / "samples" / f"epoch_{ep:03d}.png", nrow=8, normalize=True, value_range=(-1, 1))
        state = {"G": netG.state_dict(), "D": netD.state_dict(), "epoch": ep, "args": vars(args)}
        if netG_ema is not None:
            state["G_ema"] = netG_ema.state_dict()
        torch.save(state, out / "ckpt" / "last.pt")

        if ep % args.eval_every == 0 or ep == args.epochs:
            train_time = time.time() - t_start
            t0 = time.time()
            m = evaluator.evaluate(G, eval_real, args.latent_dim)
            metric_writer.writerow([ep, engine.state.iteration, f"{m['is']:.4f}", f"{m['fid']:.6f}", f"{m['fid2048']:.3f}", f"{train_time:.0f}"])
            metric_file.flush()
            logger.info(
                f"Epoch [{ep}/{args.epochs}] IS={m['is']:.3f} FID(probs)={m['fid']:.5f} "
                f"FID(pool3)={m['fid2048']:.2f} (eval {time.time() - t0:.0f}s)"
            )
            if m["fid2048"] < best["fid2048"]:
                best.update(m, epoch=ep)
                torch.save(state, out / "ckpt" / "best.pt")
        if ep % 10 == 0:
            torch.save(state, out / "ckpt" / f"epoch_{ep:03d}.pt")

    trainer.run(train_data, max_epochs=args.epochs)
    loss_file.close()
    metric_file.close()
    json.dump(best, open(out / "best.json", "w"), indent=2)
    logger.info(f"done in {(time.time() - t_start) / 60:.1f} min; best {best}")


if __name__ == "__main__":
    main()

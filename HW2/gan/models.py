import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class Generator(nn.Module):
    """DCGAN generator: z (latent_dim x 1 x 1) -> 3 x 64 x 64 image in [-1, 1]."""

    def __init__(self, latent_dim=100, ngf=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # ngf x 32 x 32
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
            # 3 x 64 x 64
        )

    def forward(self, z):
        return self.model(z)


class Discriminator(nn.Module):
    """DCGAN discriminator returning raw logits (B,).

    With ``spectral=True`` every conv is spectrally normalised and BatchNorm is
    dropped (SN-GAN style); otherwise it is the original DCGAN discriminator.
    """

    def __init__(self, ndf=64, spectral=False):
        super().__init__()
        sn = spectral_norm if spectral else (lambda m: m)
        norm = (lambda c: nn.Identity()) if spectral else nn.BatchNorm2d
        self.model = nn.Sequential(
            # 3 x 64 x 64
            sn(nn.Conv2d(3, ndf, 4, 2, 1, bias=spectral)),
            nn.LeakyReLU(0.2, inplace=True),
            # ndf x 32 x 32
            sn(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=spectral)),
            norm(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*2) x 16 x 16
            sn(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=spectral)),
            norm(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*4) x 8 x 8
            sn(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=spectral)),
            norm(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*8) x 4 x 4
            sn(nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=spectral)),
        )

    def forward(self, x):
        return self.model(x).view(-1)


def initialize_fn(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        weight = m.weight_orig if hasattr(m, "weight_orig") else m.weight
        nn.init.normal_(weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


@torch.no_grad()
def recalibrate_bn(G, latent_dim, device, n_batches=20, batch_size=256):
    """Re-estimate BatchNorm running stats for (EMA-averaged) generator weights.

    Running stats copied from the online generator do not match averaged
    weights, which makes eval-mode samples collapse to a flat colour.
    """
    bns = [m for m in G.modules() if isinstance(m, nn.BatchNorm2d)]
    saved = [m.momentum for m in bns]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None  # cumulative average
    G.train()
    for _ in range(n_batches):
        G(torch.randn(batch_size, latent_dim, 1, 1, device=device))
    for m, mom in zip(bns, saved):
        m.momentum = mom
    G.eval()
    return G


def count_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    g, d = Generator(), Discriminator()
    z = torch.randn(4, 100, 1, 1)
    x = g(z)
    print(x.shape, d(x).shape, count_params(g), count_params(d))

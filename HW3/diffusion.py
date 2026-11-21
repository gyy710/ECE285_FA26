"""A time-conditioned U-Net and variance-preserving diffusion in RGB space."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.time_affine = nn.Linear(time_dim, 2 * channels)

    def forward(self, x, time_embedding):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.time_affine(time_embedding)[:, :, None, None].chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        return x + self.conv2(F.silu(h))


class Denoiser(nn.Module):
    """U-Net with additive encoder skips and time-conditioned residual blocks."""

    def __init__(self, width=32):
        super().__init__()
        time_dim = 4 * width
        frequencies = torch.exp(-math.log(10000) * torch.arange(width // 2) / (width // 2 - 1))
        self.register_buffer("frequencies", frequencies)
        self.time_mlp = nn.Sequential(nn.Linear(width, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim), nn.SiLU())
        self.stem = nn.Conv2d(3, width, 4, stride=2, padding=1)
        self.enc1 = ResidualBlock(width, time_dim)
        self.down1 = nn.Conv2d(width, 2 * width, 4, stride=2, padding=1)
        self.enc2 = ResidualBlock(2 * width, time_dim)
        self.down2 = nn.Conv2d(2 * width, 3 * width, 4, stride=2, padding=1)
        self.mid1 = ResidualBlock(3 * width, time_dim)
        self.mid2 = ResidualBlock(3 * width, time_dim)
        self.up2 = nn.Conv2d(3 * width, 2 * width, 3, padding=1)
        self.dec2 = ResidualBlock(2 * width, time_dim)
        self.up1 = nn.Conv2d(2 * width, width, 3, padding=1)
        self.dec1 = ResidualBlock(width, time_dim)
        self.head = nn.Conv2d(width, 4 * 8, 3, padding=1)
        self.out = nn.Conv2d(8 + 3, 3, 3, padding=1)

    def forward(self, x, t):
        angles = t[:, None] * self.frequencies[None, :]
        embedding = self.time_mlp(torch.cat([angles.sin(), angles.cos()], dim=1))
        s1 = self.enc1(self.stem(x), embedding)
        s2 = self.enc2(self.down1(s1), embedding)
        h = self.mid2(self.mid1(self.down2(s2), embedding), embedding)
        h = self.dec2(s2 + self.up2(F.interpolate(h, scale_factor=2, mode="nearest")), embedding)
        h = self.dec1(s1 + self.up1(F.interpolate(h, scale_factor=2, mode="nearest")), embedding)
        h = F.pixel_shuffle(self.head(h), 2)
        return self.out(torch.cat([F.silu(h), x], dim=1))


class Diffusion(nn.Module):
    def __init__(self, steps=256):
        super().__init__()
        self.steps = steps
        boundary = torch.arange(steps + 1, dtype=torch.float64) / steps
        power = torch.cos((boundary + .008) / 1.008 * math.pi / 2).square()
        betas = (1 - power[1:] / power[:-1]).clamp(max=.999)
        cumulative = (1 - betas).cumprod(0)
        self.register_buffer("signal", cumulative.sqrt().float())
        self.register_buffer("noise", (1 - cumulative).sqrt().float())

    def coefficients(self, t):
        return self.signal[t, None, None, None], self.noise[t, None, None, None]

    def loss(self, model, clean):
        batch = clean.shape[0]
        # One uniform draw per equally spaced stratum. Shuffled training images
        # make every image position exchangeable, and every step spans noise levels.
        positions = torch.arange(batch, device=clean.device) + torch.rand(batch, device=clean.device)
        # Rounding at the upper boundary can produce exactly self.steps in float32.
        t = (positions * (self.steps / batch)).long().clamp_max(self.steps - 1)
        noise = torch.randn_like(clean)
        a, s = self.coefficients(t)
        xt = a * clean + s * noise
        target = a * noise - s * clean
        return F.mse_loss(model(xt, t), target)

    @torch.no_grad()
    def sample(self, model, latents, evaluations=48):
        """Deterministic DDIM with a clipped clean estimate and matching noise."""
        if not 1 <= evaluations <= self.steps:
            raise ValueError("Sampler steps must be between 1 and the training schedule length")
        sequence = torch.linspace(self.steps - 1, 0, evaluations).round().long().tolist()
        x = latents
        for i, step in enumerate(sequence):
            t = torch.full((x.shape[0],), step, device=x.device, dtype=torch.long)
            velocity = model(x, t)
            a, s = self.coefficients(t)
            clean = (a * x - s * velocity).clamp(-1, 1)
            if i + 1 == len(sequence):
                return clean
            noise = (x - a * clean) / s
            prev = sequence[i + 1]
            x = self.signal[prev] * clean + self.noise[prev] * noise

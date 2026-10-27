"""Step 3: A convolutional VAE for 64x64 or 128x128 grayscale chest X-rays.

Run python model.py to check the forward pass, loss, and backward pass on a small
batch of real images, without an optimizer update. Inputs come from dataset.py
and lie in [0,1]. decode/forward return logits; apply sigmoid for image display
or export.
"""

import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class ConvVAE(nn.Module):
    def __init__(self, latent_dim=128, image_size=64):
        super().__init__()
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if image_size not in (64, 128):
            raise ValueError("ConvVAE supports image_size=64 or 128 only")
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.feature_spatial = image_size // 16
        feature_dim = 256 * self.feature_spatial ** 2

        # Four spatial halvings give 4x4 features for 64 inputs, or 8x8 for 128.
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),  # [B,32,H/2,H/2], H=64 or 128
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),  # [B,64,H/4,H/4]
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),  # [B,128,H/8,H/8]
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),  # [B,256,H/16,H/16]
        )
        self.fc_mu = nn.Linear(feature_dim, latent_dim)
        self.fc_logvar = nn.Linear(feature_dim, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, feature_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),  # [B,128,H/8,H/8]
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),  # [B,64,H/4,H/4]
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),  # [B,32,H/2,H/2]
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            # [B,1,H,H]: retain logits for BCE-with-logits.
        )

    def encode(self, images):
        """Return the diagonal Gaussian q(z|x) mean and log variance, [B,latent_dim]."""
        if images.ndim != 4 or tuple(images.shape[1:]) != (1, self.image_size, self.image_size):
            raise ValueError(
                f"ConvVAE expects input [B,1,{self.image_size},{self.image_size}], got {tuple(images.shape)}."
            )
        features = self.encoder(images).flatten(start_dim=1)
        return self.fc_mu(features), self.fc_logvar(features)

    @staticmethod
    def reparameterize(mu, logvar):
        """Sample randomness through epsilon while preserving mu and logvar gradients."""
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mu + std * epsilon

    def decode(self, z):
        """Decode [B,latent_dim] into reconstruction logits, before sigmoid."""
        features = F.relu(self.decoder_input(z))
        features = features.reshape(z.shape[0], 256, self.feature_spatial, self.feature_spatial)
        return self.decoder(features)

    def forward(self, images):
        """Return (logits, mu, logvar); z is sampled even in eval mode."""
        mu, logvar = self.encode(images)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def build_model(config):
    """Build the configured model after checking compatible image sizes and channels."""
    if config["data"]["channels"] != 1 or config["data"]["image_size"] not in (64, 128):
        raise ValueError("ConvVAE supports 64x64 or 128x128 single-channel images; use image_size=64 or 128 and channels=1.")
    return ConvVAE(latent_dim=config["model"]["latent_dim"], image_size=config["data"]["image_size"])


def vae_loss(logits, images, mu, logvar, beta=1.0, reconstruction_scale=1.0):
    """Return total, reconstruction, kl, using per-image sums and batch means.

    This baseline uses BCE with grayscale values in [0,1] as soft targets.
    reconstruction_scale scales only reconstruction; 0.25 for 128 inputs
    matches the reference pixel count of 64 inputs.
    KL compares q(z|x)=N(mu, diag(exp(logvar))) with the prior N(0,I).
    """
    if not math.isfinite(reconstruction_scale) or reconstruction_scale <= 0:
        raise ValueError("reconstruction_scale must be finite and positive")
    pixel_losses = F.binary_cross_entropy_with_logits(logits, images, reduction="none")
    reconstruction = pixel_losses.flatten(start_dim=1).sum(dim=1).mean() * reconstruction_scale
    kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(dim=1).mean()
    total = reconstruction + beta * kl
    return total, reconstruction, kl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config.json")
    parser.add_argument("--device", default=None, help="Use the configured device (mps by default), or specify cpu.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for the model check; does not change the configuration file.")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive.")
    config = json.loads(args.config.read_text())
    config["training"]["batch_size"] = args.batch_size
    device = torch.device(args.device or config["training"]["device"])
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; check the environment or use --device cpu for the model check.")
    torch.manual_seed(config["seed"])

    from dataset import build_dataloaders

    model = build_model(config).to(device)
    images, _ = next(iter(build_dataloaders(config)["train"]))
    images = images.to(device)
    logits, mu, logvar = model(images)
    total, reconstruction, kl = vae_loss(
        logits, images, mu, logvar, beta=config["training"]["beta"],
        reconstruction_scale=config["training"].get("reconstruction_scale", 1.0),
    )
    if not torch.isfinite(total).item():
        raise RuntimeError("The model check found a non-finite loss.")
    total.backward()
    for name, parameter in model.named_parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all().item():
            raise RuntimeError(f"Parameter {name} has a missing or non-finite gradient.")

    print(f"Device: {device}; parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Input / logits: {tuple(images.shape)} / {tuple(logits.shape)}")
    print(f"mu / logvar: {tuple(mu.shape)} / {tuple(logvar.shape)}")
    print(f"Loss: total={total.item():.3f}, reconstruction={reconstruction.item():.3f}, KL={kl.item():.3f}")
    print("Backward check passed: all model parameters have finite gradients.")
    print("This is an untrained model check; no optimizer update was performed.")


if __name__ == "__main__":
    main()

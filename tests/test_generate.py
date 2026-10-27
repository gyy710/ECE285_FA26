"""Checks for seeded prior generation and evidence-backed image exports."""

import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
import torch
from torch import nn

import generate
from dataset import build_dataloaders, save_split_manifest
from model import build_model


class TinyVAE(nn.Module):
    latent_dim = 2

    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.1))

    def decode(self, z):
        return (z[:, :1] + self.bias).reshape(-1, 1, 1, 1).expand(-1, 1, 4, 4)

    def encode(self, images):
        mu = images.mean((1, 2, 3)).unsqueeze(1).repeat(1, self.latent_dim)
        return mu, torch.zeros_like(mu)

    def forward(self, images):
        raise AssertionError("Deterministic reconstruction should explicitly decode the mean.")


class GenerationTests(unittest.TestCase):
    def test_seeded_prior_is_repeatable_and_independent_of_batch_size(self):
        model = TinyVAE()
        first = generate.generate_samples(model, 5, 123, 2, torch.device("cpu"))
        second = generate.generate_samples(model, 5, 123, 5, torch.device("cpu"))
        other_seed = generate.generate_samples(model, 5, 124, 2, torch.device("cpu"))
        torch.testing.assert_close(first, second)
        self.assertFalse(torch.equal(first, other_seed))
        self.assertEqual(tuple(first.shape), (5, 1, 4, 4))
        self.assertEqual(first.device.type, "cpu")
        self.assertFalse(first.requires_grad)
        self.assertGreaterEqual(first.min().item(), 0.0)
        self.assertLessEqual(first.max().item(), 1.0)

    def test_generation_and_reconstruction_leave_global_rng_and_parameters_alone(self):
        model = TinyVAE()
        before_rng = torch.get_rng_state().clone()
        before_weight = model.bias.detach().clone()
        generate.generate_samples(model, 3, 123, 2, torch.device("cpu"))
        images = torch.full((2, 1, 4, 4), 0.4)
        reconstruction = generate.reconstruct_samples(model, images, torch.device("cpu"))
        torch.testing.assert_close(reconstruction, torch.full_like(images, torch.sigmoid(torch.tensor(0.5)).item()))
        self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))
        torch.testing.assert_close(before_weight, model.bias)

    def test_nonpositive_counts_are_rejected(self):
        for count, batch in ((0, 2), (2, 0)):
            with self.assertRaises(ValueError):
                generate.generate_samples(TinyVAE(), count, 123, batch, torch.device("cpu"))


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        data_root = self.root / "data"
        for split in ("train", "val", "test"):
            for label in ("NORMAL", "PNEUMONIA"):
                folder = data_root / split / label
                folder.mkdir(parents=True)
                Image.new("L", (12, 10), 100).save(folder / f"{split}_{label}.png")
        self.config = {
            "seed": 42,
            "data": {"root": str(data_root), "image_size": 64, "channels": 1, "num_workers": 0},
            "model": {"latent_dim": 4},
            "training": {"batch_size": 2, "device": "cpu"},
            "generation": {"seed": 123, "grid_samples": 64},
        }
        manifest = save_split_manifest(build_dataloaders(self.config), self.config, self.root / "original_manifest.json")
        self.checkpoint = self.root / "best.pt"
        torch.save({
            "epoch": 7,
            "config": self.config,
            "model_state_dict": build_model(self.config).state_dict(),
            "split_fingerprint": manifest["fingerprint"],
        }, self.checkpoint)

    def test_export_uses_checkpoint_config_and_saves_distinct_evidence(self):
        destination = self.root / "generated"
        generate.main([
            "--checkpoint", str(self.checkpoint), "--output-dir", str(destination),
            "--num-samples", "3", "--batch-size", "2", "--seed", "321",
            "--device", "cpu", "--reconstruct",
        ])
        images = sorted((destination / "samples").glob("*.png"))
        self.assertEqual(len(images), 3)
        for path in images:
            with Image.open(path) as image:
                self.assertEqual(image.size, (64, 64))
                self.assertEqual(image.mode, "L")
        self.assertTrue((destination / "generated_grid.png").is_file())
        self.assertTrue((destination / "reconstruction_grid.png").is_file())
        metadata = json.loads((destination / "metadata.json").read_text())
        self.assertEqual(metadata["checkpoint_epoch"], 7)
        self.assertEqual(metadata["latent_dim"], 4)
        self.assertEqual(metadata["seed"], 321)
        self.assertEqual(metadata["num_samples"], 3)
        self.assertEqual(len(metadata["checkpoint_sha256"]), 64)
        self.assertTrue(all(path.startswith("val/") for path in metadata["reconstruction_source_files"]))
        with self.assertRaises(FileExistsError):
            generate.main(["--checkpoint", str(self.checkpoint), "--output-dir", str(destination)])

    def test_prior_generation_does_not_require_the_training_dataset(self):
        checkpoint = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        checkpoint["config"]["data"]["root"] = str(self.root / "missing_data")
        torch.save(checkpoint, self.checkpoint)
        destination = self.root / "prior_only"
        generate.main([
            "--checkpoint", str(self.checkpoint), "--output-dir", str(destination),
            "--num-samples", "1", "--device", "cpu",
        ])
        self.assertTrue((destination / "generated_grid.png").is_file())
        self.assertFalse((destination / "reconstruction_grid.png").exists())

    def test_reconstruction_rejects_a_changed_split_before_writing_exports(self):
        checkpoint = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        checkpoint["split_fingerprint"] = "changed"
        torch.save(checkpoint, self.checkpoint)
        destination = self.root / "wrong_split"
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            generate.main([
                "--checkpoint", str(self.checkpoint), "--output-dir", str(destination),
                "--num-samples", "1", "--device", "cpu", "--reconstruct",
            ])
        self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()

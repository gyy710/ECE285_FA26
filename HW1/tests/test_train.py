"""Synthetic CPU checks: epoch accounting, saved artifacts and exact resume."""

import copy
import csv
import contextlib
import io
import shutil
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import train


class TinyVAE(nn.Module):
    """Small real VAE with the same interface; keep trainer tests inexpensive."""
    latent_dim = 2

    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(4, 4)
        self.decoder = nn.Linear(2, 4)

    def encode(self, images):
        mu, logvar = self.encoder(images.flatten(1)).chunk(2, dim=1)
        return mu, logvar

    def decode(self, z):
        return self.decoder(z).reshape(-1, 1, 2, 2)

    def forward(self, images):
        mu, logvar = self.encode(images)
        return self.decode(mu + (0.5 * logvar).exp() * torch.randn_like(mu)), mu, logvar


class ConstantVAE(TinyVAE):
    """Opposing train/val intensities make later validation strictly worse."""
    def __init__(self):
        nn.Module.__init__(self)
        self.bias = nn.Parameter(torch.tensor(0.0))

    def encode(self, images):
        zeros = images.new_zeros((len(images), 2))
        return zeros, zeros

    def decode(self, z):
        return self.bias.expand(len(z), 1, 2, 2)


class ZeroDecoder(TinyVAE):
    def encode(self, images):
        mu = images.flatten(1)[:, :1]
        return mu, torch.zeros_like(mu)

    def decode(self, z):
        return torch.zeros(len(z), 1, 2, 2)


def synthetic_loader():
    images = torch.linspace(0.05, 0.95, 20).reshape(5, 1, 2, 2)
    return DataLoader(TensorDataset(images, torch.zeros(5)), batch_size=2,
                      generator=torch.Generator().manual_seed(17))


class EpochTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(callable(getattr(train, "run_epoch", None)), "run_epoch must be implemented")

    def test_metrics_weight_short_last_batch_by_samples(self):
        loader = synthetic_loader()
        result = train.run_epoch(ZeroDecoder(), loader, "cpu", beta=0.5)
        # Every image has four zero logits: BCE = 4*log(2); KL = mu^2/2.
        means = loader.dataset.tensors[0].flatten(1)[:, 0]
        expected_kl = 0.5 * means.square().mean().item()
        self.assertEqual(result["samples"], 5)
        self.assertAlmostEqual(result["reconstruction"], 2.772588722, places=6)
        self.assertAlmostEqual(result["kl"], expected_kl, places=6)
        self.assertAlmostEqual(result["total"], 2.772588722 + 0.5 * expected_kl, places=6)

    def test_reconstruction_scale_changes_only_reconstruction_not_kl(self):
        original = train.run_epoch(ZeroDecoder(), synthetic_loader(), "cpu", beta=0.5)
        scaled = train.run_epoch(
            ZeroDecoder(), synthetic_loader(), "cpu", beta=0.5, reconstruction_scale=0.25
        )
        self.assertEqual(scaled["samples"], 5)
        self.assertAlmostEqual(scaled["reconstruction"], original["reconstruction"] / 4, places=6)
        self.assertAlmostEqual(scaled["kl"], original["kl"], places=6)
        self.assertAlmostEqual(scaled["total"], original["reconstruction"] / 4 + 0.5 * original["kl"], places=6)

    def test_validation_is_repeatable_and_leaves_parameters_and_global_rng_alone(self):
        model = TinyVAE()
        before = copy.deepcopy(model.state_dict())
        rng = torch.get_rng_state().clone()
        first = train.run_epoch(model, synthetic_loader(), "cpu", beta=1, validation_seed=123)
        second = train.run_epoch(model, synthetic_loader(), "cpu", beta=1, validation_seed=123)
        self.assertEqual(first, second)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertFalse(model.training)
        for name, parameter in model.state_dict().items():
            self.assertTrue(torch.equal(parameter, before[name]))
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        different = train.run_epoch(model, synthetic_loader(), "cpu", beta=1, validation_seed=124)
        self.assertNotEqual(first["reconstruction"], different["reconstruction"])

    def test_training_updates_parameters(self):
        model = TinyVAE()
        before = copy.deepcopy(model.state_dict())
        result = train.run_epoch(model, synthetic_loader(), "cpu", beta=1,
                                 optimizer=torch.optim.Adam(model.parameters(), lr=0.01))
        self.assertEqual(result["samples"], 5)
        self.assertTrue(model.training)
        self.assertTrue(any(not torch.equal(value, before[name])
                            for name, value in model.state_dict().items()))

    def test_previews_use_fixed_prior_without_consuming_training_rng(self):
        model = TinyVAE()
        config = {"model": {"latent_dim": 2}, "generation": {"seed": 123, "grid_samples": 4}}
        before = torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            train.save_previews(model, synthetic_loader().dataset, "cpu", config, output, 1)
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            torch.randn(17)  # Advancing global RNG must not change preview samples.
            train.save_previews(model, synthetic_loader().dataset, "cpu", config, output, 2)
            for prefix in ("generated", "reconstruction"):
                self.assertEqual((output / f"{prefix}_epoch_001.png").read_bytes(),
                                 (output / f"{prefix}_epoch_002.png").read_bytes())
        self.assertFalse(model.training)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_nonfinite_loss_and_empty_loader_are_rejected(self):
        model = TinyVAE()
        with torch.no_grad():
            model.encoder.weight.fill_(float("nan"))
        with self.assertRaisesRegex(RuntimeError, "finite"):
            train.run_epoch(model, synthetic_loader(), "cpu", beta=1)
        empty = DataLoader(TensorDataset(torch.zeros(0, 1, 2, 2)), batch_size=2,
                           generator=torch.Generator().manual_seed(1))
        with self.assertRaisesRegex(ValueError, "empty|samples"):
            train.run_epoch(TinyVAE(), empty, "cpu", beta=1)


class TrainingArtifactsTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(callable(getattr(train, "run_epoch", None)), "trainer must be implemented")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        data = self.root / "images"
        for split in ("train", "val", "test"):
            for label in ("NORMAL", "PNEUMONIA"):
                folder = data / split / label
                folder.mkdir(parents=True)
                for index in range(3):
                    Image.new("L", (2, 2), color=40 + 30 * index).save(folder / f"image{index}.png")
        self.config = {
            "seed": 42,
            "data": {"root": str(data), "channels": 1, "image_size": 2, "num_workers": 0},
            "model": {"latent_dim": 2},
            "training": {"device": "cpu", "batch_size": 4, "epochs": 2,
                         "optimizer": "adam", "learning_rate": 0.001, "beta": 1.0,
                         "reconstruction_loss": "bce_with_logits", "output_dir": str(self.root / "unused")},
            "generation": {"seed": 123, "grid_samples": 4},
        }
        self.config_path = self.root / "config.json"
        self.config_path.write_text(json.dumps(self.config))

    def invoke(self, output, epochs, resume=None):
        args = ["--config", str(self.config_path), "--output-dir", str(output),
                "--epochs", str(epochs), "--device", "cpu"]
        if resume:
            args += ["--resume", str(resume)]
        # Only replace the large network; loader, loss, optimizer, plotting and IO stay real.
        with patch("train.build_model", side_effect=lambda config: getattr(self, "model_factory", TinyVAE)()):
            train.main(args)

    def test_artifacts_are_readable_and_existing_run_is_protected(self):
        output = self.root / "smoke"
        self.invoke(output, 1)
        expected = ("config.json", "environment.json", "split_manifest.json", "history.csv",
                    "loss_curves.png", "best.pt", "last.pt", "generated_epoch_001.png",
                    "reconstruction_epoch_001.png")
        for filename in expected:
            self.assertTrue((output / filename).is_file(), filename)
        checkpoint = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
        best = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
        manifest = json.loads((output / "split_manifest.json").read_text())
        self.assertEqual(checkpoint["epoch"], 1)
        self.assertEqual(checkpoint["split_fingerprint"], manifest["fingerprint"])
        self.assertEqual(checkpoint["config"]["training"]["epochs"], 1)
        self.assertEqual(best["best_val_loss"], checkpoint["history"][0]["val_total"])
        self.assertTrue(checkpoint["optimizer_state_dict"]["state"])
        with (output / "history.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 1)
        self.assertEqual(int(rows[0]["train_samples"]), 6)
        self.assertEqual(int(rows[0]["val_samples"]), 6)
        self.assertGreater(float(rows[0]["elapsed_seconds"]), 0)
        with Image.open(output / "reconstruction_epoch_001.png") as image:
            self.assertEqual(image.height, 10)
        previous = (output / "last.pt").read_bytes()
        with self.assertRaisesRegex((ValueError, FileExistsError), "resume|exist|overwrite"):
            self.invoke(output, 1)
        self.assertEqual(previous, (output / "last.pt").read_bytes())

    def test_configured_scale_is_used_in_recorded_validation_and_saved(self):
        self.config["training"]["reconstruction_scale"] = 0.25
        self.config_path.write_text(json.dumps(self.config))
        self.model_factory = ConstantVAE
        output = self.root / "scaled"
        self.invoke(output, 1)
        checkpoint = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
        self.assertEqual(checkpoint["config"]["training"]["reconstruction_scale"], 0.25)
        bias = checkpoint["model_state_dict"]["bias"]
        targets = torch.tensor([40, 70, 100, 40, 70, 100], dtype=torch.float32) / 255
        expected = torch.nn.functional.binary_cross_entropy_with_logits(
            bias.expand_as(targets), targets, reduction="mean"
        ).item() * 4 * 0.25
        self.assertAlmostEqual(checkpoint["history"][-1]["val_reconstruction"], expected, places=6)
        self.assertEqual(checkpoint["history"][-1]["val_kl"], 0.0)

    def test_invalid_reconstruction_scale_is_rejected_before_writing(self):
        for scale in (0.0, -0.25, float("nan")):
            self.config["training"]["reconstruction_scale"] = scale
            self.config_path.write_text(json.dumps(self.config))
            output = self.root / "invalid_scale"
            with self.assertRaisesRegex(ValueError, "reconstruction_scale"):
                self.invoke(output, 1)
            self.assertFalse(output.exists())

    def test_resumed_cpu_run_matches_uninterrupted_and_carries_history_to_new_directory(self):
        full, first, resumed = (self.root / name for name in ("full", "first", "resumed"))
        self.invoke(full, 2)
        self.invoke(first, 1)
        self.invoke(resumed, 2, first / "last.pt")
        expected = torch.load(full / "last.pt", map_location="cpu", weights_only=True)
        actual = torch.load(resumed / "last.pt", map_location="cpu", weights_only=True)
        self.assertEqual(actual["epoch"], 2)
        self.assertEqual(len(actual["history"]), 2)
        for name, tensor in expected["model_state_dict"].items():
            self.assertTrue(torch.equal(tensor, actual["model_state_dict"][name]), name)
        for left, right in zip(expected["history"], actual["history"]):
            for key in ("train_total", "val_total", "train_reconstruction", "val_kl"):
                self.assertEqual(left[key], right[key], key)
        self.assertTrue(torch.equal(expected["rng_states"]["torch_cpu"], actual["rng_states"]["torch_cpu"]))
        self.assertTrue((resumed / "best.pt").is_file())

    def test_moving_last_checkpoint_alone_retains_earlier_best_model(self):
        self.model_factory = ConstantVAE
        data = Path(self.config["data"]["root"])
        for split, intensity in (("train", 20), ("val", 230)):
            for path in (data / split).rglob("*.png"):
                Image.new("L", (2, 2), color=intensity).save(path)
        first = self.root / "first"
        self.invoke(first, 2)
        best = torch.load(first / "best.pt", map_location="cpu", weights_only=True)
        self.assertEqual(best["epoch"], 1)
        isolated = self.root / "isolated.pt"
        shutil.copyfile(first / "last.pt", isolated)
        shutil.rmtree(first)
        resumed = self.root / "resumed"
        self.invoke(resumed, 3, isolated)
        retained = torch.load(resumed / "best.pt", map_location="cpu", weights_only=True)
        self.assertEqual(retained["epoch"], 1)
        self.assertTrue(torch.equal(best["model_state_dict"]["bias"], retained["model_state_dict"]["bias"]))
        self.assertEqual(len(list(csv.DictReader((resumed / "history.csv").read_text().splitlines()))), 3)

    def test_resume_rejects_changed_config_or_split_and_no_remaining_epochs(self):
        output = self.root / "first"
        self.invoke(output, 1)
        self.config["training"]["beta"] = 0.5
        self.config_path.write_text(json.dumps(self.config))
        with self.assertRaisesRegex(ValueError, "config|compatible"):
            self.invoke(self.root / "bad_config", 2, output / "last.pt")
        self.config["training"]["beta"] = 1.0
        self.config_path.write_text(json.dumps(self.config))
        with self.assertRaisesRegex(ValueError, "epoch|remaining"):
            self.invoke(self.root / "finished", 1, output / "last.pt")
        added = Path(self.config["data"]["root"]) / "train" / "NORMAL" / "extra.png"
        Image.new("L", (2, 2), color=80).save(added)
        with self.assertRaisesRegex(ValueError, "split|fingerprint"):
            self.invoke(self.root / "bad_split", 2, output / "last.pt")

    def test_nonpositive_epochs_are_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.invoke(self.root / "bad", 0)


if __name__ == "__main__":
    unittest.main()

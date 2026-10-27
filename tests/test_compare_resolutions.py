"""Synthetic, matched-source checks for the 64 versus 128 comparison."""

import copy
import hashlib
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import generate
from dataset import build_dataloaders, save_split_manifest


class TinyVAE(nn.Module):
    latent_dim = 2

    def __init__(self, size):
        super().__init__()
        self.image_size = size
        self.bias = nn.Parameter(torch.tensor(0.1))

    def encode(self, images):
        assert images.shape[2:] == (self.image_size, self.image_size)
        mu = images.mean((1, 2, 3)).unsqueeze(1).repeat(1, 2)
        return mu, torch.zeros_like(mu)

    def decode(self, z):
        return (z[:, :1] + self.bias).reshape(-1, 1, 1, 1).expand(-1, 1, self.image_size, self.image_size)


class KnownDataset(Dataset):
    classes = ["NORMAL", "PNEUMONIA"]
    class_to_idx = {"NORMAL": 0, "PNEUMONIA": 1}

    def __init__(self, size):
        self.size = size
        self.samples = [(f"/data/val/NORMAL/{index}.png", 0) for index in range(5)]

    def __len__(self):
        return 5

    def __getitem__(self, index):
        return torch.full((1, self.size, self.size), index / 4), 0


class ResolutionComparisonTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("compare_resolutions"), "Resolution comparison CLI must exist.")
        self.compare = importlib.import_module("compare_resolutions")
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data_root = self.root / "data"
        for split in ("train", "val", "test"):
            for label in ("NORMAL", "PNEUMONIA"):
                folder = self.data_root / split / label
                folder.mkdir(parents=True)
                for index in range(3):
                    Image.new("L", (10, 12), 30 + index * 50).save(folder / f"{index}.png")
        config = {
            "seed": 42,
            "data": {"root": str(self.data_root), "image_size": 64, "channels": 1, "num_workers": 0},
            "model": {"latent_dim": 2},
            "training": {"batch_size": 4, "device": "mps", "epochs": 2, "beta": 0.1,
                         "output_dir": "unused", "learning_rate": 0.001, "optimizer": "adam"},
            "generation": {"seed": 123, "grid_samples": 64},
        }
        manifest = save_split_manifest(build_dataloaders(config), config, self.root / "manifest.json")
        self.baseline = {
            "epoch": 2, "config": config, "split_fingerprint": manifest["fingerprint"],
            "model_state_dict": TinyVAE(64).state_dict(),
            "history": [{"epoch": 1, "val_reconstruction": 110.0, "val_kl": 3.0},
                        {"epoch": 2, "val_reconstruction": 100.0, "val_kl": 2.0}],
        }
        self.experiment = copy.deepcopy(self.baseline)
        self.experiment["config"]["data"]["image_size"] = 128
        self.experiment["config"]["training"].update(reconstruction_scale=0.25, epochs=3, output_dir="variant")
        self.experiment["history"][-1].update(val_reconstruction=90.0, val_kl=10.0)
        self.paths = [self.root / "baseline.pt", self.root / "experiment.pt"]
        self.output = self.root / "comparison"
        self.model_patch = patch.object(generate, "build_model", lambda config: TinyVAE(config["data"]["image_size"]))
        self.model_patch.start()
        self.addCleanup(self.model_patch.stop)

    def run_comparison(self):
        for checkpoint, path in zip((self.baseline, self.experiment), self.paths):
            torch.save(checkpoint, path)
        return self.compare.main([
            "--baseline", str(self.paths[0]), "--experiment", str(self.paths[1]),
            "--output-dir", str(self.output), "--device", "cpu",
        ])

    def test_checks_matched_epochs_configs_scales_and_finite_contiguous_history(self):
        changes = [
            ("epoch", lambda cp: cp.update(epoch=1)),
            ("fingerprint", lambda cp: cp.update(split_fingerprint="changed")),
            ("config", lambda cp: cp["config"]["training"].update(beta=1)),
            ("config", lambda cp: cp["config"]["training"].update(learning_rate=0.01)),
            ("config", lambda cp: cp["config"]["model"].update(latent_dim=3)),
            ("config", lambda cp: cp["config"].update(seed=43)),
            ("config", lambda cp: cp["config"]["generation"].update(seed=124)),
            ("64.*128", lambda cp: cp["config"]["data"].update(image_size=64)),
            ("reference", lambda cp: cp["config"]["training"].update(reconstruction_scale=1)),
            ("scale", lambda cp: cp["config"]["training"].update(reconstruction_scale=float("nan"))),
            ("scale", lambda cp: cp["config"]["training"].update(reconstruction_scale=0)),
            ("history", lambda cp: cp["history"].pop(0)),
            ("finite", lambda cp: cp["history"][0].update(val_kl=float("inf"))),
            ("finite", lambda cp: cp["history"][0].update(train_total=float("nan"))),
        ]
        self.compare._check_compatibility(self.baseline, self.experiment)
        for message, change in changes:
            with self.subTest(message=message):
                changed = copy.deepcopy(self.experiment)
                change(changed)
                with self.assertRaisesRegex(ValueError, message):
                    self.compare._check_compatibility(self.baseline, changed)

    def test_all_validation_metrics_weight_samples_and_short_last_batch(self):
        loaders = [DataLoader(KnownDataset(size), batch_size=2) for size in (64, 128)]
        models = [TinyVAE(size) for size in (64, 128)]
        def known_reconstruction(model, images, device):
            # Baseline predicts zero; experiment predicts one for each of five targets.
            return torch.full_like(images, 0 if model.image_size == 64 else 1)
        with patch.object(generate, "reconstruct_samples", known_reconstruction):
            metrics = self.compare.evaluate_common_target(models, loaders, torch.device("cpu"))
        self.assertEqual(metrics["sample_count"], 5)
        self.assertEqual(metrics["pixel_count"], 5 * 128 * 128)
        self.assertEqual(metrics["target_image_size"], 128)
        # Independent sums: targets 0, .25, .5, .75, 1 -> MAE .5, MSE .375.
        for role in ("baseline", "experiment"):
            self.assertAlmostEqual(metrics[role]["mae"], 0.5)
            self.assertAlmostEqual(metrics[role]["mse"], 0.375)
        self.assertEqual(metrics["relative_change_percent"], {"mae": 0.0, "mse": 0.0})

    def test_rejects_different_validation_source_or_label_order(self):
        datasets = [KnownDataset(size) for size in (64, 128)]
        datasets[1].samples.reverse()
        with self.assertRaisesRegex(ValueError, "source|order"):
            self.compare.evaluate_common_target([TinyVAE(64), TinyVAE(128)],
                                               [DataLoader(data, batch_size=2) for data in datasets], "cpu")
        datasets[1].samples = list(datasets[0].samples)
        datasets[1].samples[0] = (datasets[1].samples[0][0], 1)
        with self.assertRaisesRegex(ValueError, "source|order"):
            self.compare.evaluate_common_target([TinyVAE(64), TinyVAE(128)],
                                               [DataLoader(data, batch_size=2) for data in datasets], "cpu")

    def test_exports_native_previews_provenance_and_all_val_metrics_without_test_reads(self):
        opened = []
        real_open = Image.open
        def validation_only(path, *args, **kwargs):
            filename = path.name if hasattr(path, "read") else path
            self.assertNotIn("test", Path(filename).parts)
            opened.append(str(filename))
            return real_open(path, *args, **kwargs)
        titles = []
        from matplotlib.axes import Axes
        real_title = Axes.set_title
        def record_title(axis, label, *args, **kwargs):
            titles.append(label)
            return real_title(axis, label, *args, **kwargs)
        with patch.object(Image, "open", validation_only), patch.object(Axes, "set_title", record_title):
            summary = self.run_comparison()
        self.assertEqual(summary["reconstruction_source_files"], [
            "val/NORMAL/0.png", "val/NORMAL/1.png", "val/PNEUMONIA/0.png", "val/PNEUMONIA/1.png",
        ])
        self.assertEqual(summary["common_target_validation"]["sample_count"], 6)
        self.assertEqual(summary["split"]["counts"], {"train": 6, "val": 6, "test": 6})
        self.assertEqual(summary["split"]["splits"]["val"][0]["path"], "val/NORMAL/0.png")
        self.assertEqual(summary["generation"]["seed"], 123)
        self.assertEqual(summary["generation"]["count_per_model"], 64)
        self.assertEqual(summary["baseline"]["generation_tensor_shape"], [64, 1, 64, 64])
        self.assertEqual(summary["experiment"]["generation_tensor_shape"], [64, 1, 128, 128])
        self.assertEqual(summary["baseline"]["reconstruction_scale"], 1)
        self.assertEqual(summary["experiment"]["reconstruction_scale"], .25)
        self.assertEqual(summary["baseline"]["reference_pixels"], 4096)
        self.assertEqual(summary["experiment"]["reference_pixels"], 4096)
        self.assertEqual(summary["baseline"]["validation"]["reconstruction_per_native_pixel"], 100 / 4096)
        self.assertEqual(summary["experiment"]["validation"]["reconstruction_per_native_pixel"], 90 / 4096)
        self.assertIn("fixed-noise", summary["validation_metric_method"])
        self.assertIn("posterior mean", summary["reconstruction_method"])
        self.assertIn("align_corners=False", summary["interpolation_method"])
        self.assertIn("capacity", summary["capacity_caveat"])
        self.assertFalse(summary["automatic_quality_winner_selection"])
        self.assertTrue(any("native 64" in title for title in titles))
        self.assertTrue(any("native 128" in title for title in titles))
        for role, path in zip(("baseline", "experiment"), self.paths):
            self.assertEqual(summary[role]["checkpoint_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(summary[role]["checkpoint_epoch"], 2)
            self.assertEqual(summary[role]["parameter_count"], 1)
        self.assertEqual(summary, json.loads((self.output / "summary.json").read_text()))
        for name in ("reconstruction_comparison.png", "generation_comparison.png", "validation_curves.png"):
            with Image.open(self.output / name) as image:
                self.assertGreater(image.width, 300)
                self.assertGreater(image.height, 200)
        self.assertGreaterEqual(len(opened), 12)

    def test_both_current_memberships_are_checked_before_inference_or_export(self):
        real_build = self.compare.build_dataloaders
        calls = []
        def altered_experiment(config):
            calls.append(config["data"]["image_size"])
            loaders = real_build(config)
            if config["data"]["image_size"] == 128:
                loaders["val"].dataset.samples.pop()
            return loaders
        with patch.object(self.compare, "build_dataloaders", altered_experiment), patch.object(generate, "reconstruct_samples") as reconstruction:
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                self.run_comparison()
            reconstruction.assert_not_called()
        self.assertEqual(calls, [64, 128])
        self.assertFalse(self.output.exists())

    def test_changed_actual_membership_rejected(self):
        Image.new("L", (10, 12), 100).save(self.data_root / "val/NORMAL/extra.png")
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            self.run_comparison()
        self.assertFalse(self.output.exists())

    def test_nonempty_output_is_preserved(self):
        self.output.mkdir()
        marker = self.output / "keep.txt"
        marker.write_text("keep")
        with self.assertRaises(FileExistsError):
            self.run_comparison()
        self.assertEqual(list(self.output.iterdir()), [marker])
        self.assertEqual(marker.read_text(), "keep")


if __name__ == "__main__":
    unittest.main()

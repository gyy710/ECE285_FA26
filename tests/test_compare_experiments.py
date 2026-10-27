"""Comparison exports use matched checkpoints and fixed validation/prior examples."""

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

import generate
from dataset import build_dataloaders, save_split_manifest


class TinyVAE(nn.Module):
    latent_dim = 2

    def __init__(self, image_size=64):
        super().__init__()
        self.image_size = image_size
        self.bias = nn.Parameter(torch.tensor(0.1))

    def encode(self, images):
        mu = images.mean((1, 2, 3)).unsqueeze(1).repeat(1, 2)
        return mu, torch.zeros_like(mu)

    def decode(self, z):
        return (z[:, :1] + self.bias).reshape(-1, 1, 1, 1).expand(-1, 1, self.image_size, self.image_size)


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("compare_experiments"), "Comparison CLI must exist.")
        self.compare = importlib.import_module("compare_experiments")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
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
            "training": {"batch_size": 2, "device": "mps", "epochs": 2, "beta": 1.0,
                         "output_dir": "unused", "learning_rate": 0.001, "optimizer": "adam",
                         "reconstruction_loss": "bce_with_logits"},
            "generation": {"seed": 123, "grid_samples": 64},
        }
        manifest = save_split_manifest(build_dataloaders(config), config, self.root / "manifest.json")
        self.baseline = {
            "epoch": 2, "config": config, "split_fingerprint": manifest["fingerprint"],
            "model_state_dict": TinyVAE().state_dict(),
            "history": [{"epoch": 1, "val_reconstruction": 110.0, "val_kl": 3.0, "val_total": 113.0},
                        {"epoch": 2, "val_reconstruction": 100.0, "val_kl": 2.0, "val_total": 102.0}],
        }
        self.experiment = copy.deepcopy(self.baseline)
        self.experiment["config"]["training"].update(beta=0.1, epochs=3, output_dir="unused_variant")
        self.experiment["history"][-1].update(val_reconstruction=90.0, val_kl=10.0, val_total=91.0)
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

    def test_exports_match_sources_and_record_metrics_without_inferred_winner(self):
        opened = []
        real_open = Image.open
        def open_without_test_images(path, *args, **kwargs):
            filename = path.name if hasattr(path, "read") else path
            self.assertNotIn("test", Path(filename).parts)
            opened.append(str(filename))
            return real_open(path, *args, **kwargs)
        generated = []
        reconstructed_inputs = []
        real_generate = generate.generate_samples
        real_reconstruct = generate.reconstruct_samples
        def recorded_generation(model, count, seed, batch_size, device):
            images = real_generate(model, count, seed, batch_size, device)
            generated.append((count, seed, images))
            return images
        def recorded_reconstruction(model, images, device):
            reconstructed_inputs.append(images.clone())
            return real_reconstruct(model, images, device)
        with patch.object(Image, "open", open_without_test_images), patch.object(generate, "generate_samples", recorded_generation), patch.object(generate, "reconstruct_samples", recorded_reconstruction):
            summary = self.run_comparison()
        self.assertEqual(len(opened), 4)
        self.assertEqual(summary["reconstruction_source_files"], [
            "val/NORMAL/0.png", "val/NORMAL/1.png", "val/PNEUMONIA/0.png", "val/PNEUMONIA/1.png",
        ])
        self.assertEqual(summary["epoch"], 2)
        self.assertEqual(summary["generation"], {"seed": 123, "count_per_model": 64, "prior": "standard_normal_N(0,I)"})
        self.assertEqual([(count, seed) for count, seed, _ in generated], [(64, 123), (64, 123)])
        torch.testing.assert_close(generated[0][2], generated[1][2])
        self.assertEqual(len(reconstructed_inputs), 2)
        torch.testing.assert_close(reconstructed_inputs[0], reconstructed_inputs[1])
        torch.testing.assert_close(reconstructed_inputs[0][:, 0, 0, 0], torch.tensor([30, 80, 30, 80]) / 255)
        self.assertEqual(summary["split"]["counts"], {"train": 6, "val": 6, "test": 6})
        self.assertEqual(summary["baseline"]["validation"], {"reconstruction": 100.0, "kl": 2.0})
        self.assertEqual(summary["experiment"]["validation"], {"reconstruction": 90.0, "kl": 10.0})
        for role in ("baseline", "experiment"):
            self.assertEqual(summary[role]["image_size"], 64)
            self.assertEqual(summary[role]["reconstruction_scale"], 1.0)
        self.assertEqual(summary["reconstruction_loss_percent_change"], -10.0)
        self.assertFalse(summary["automatic_quality_winner_selection"])
        self.assertIn("incomparable", summary["total_loss_note"])
        self.assertIn("fixed-noise", summary["validation_metric_method"])
        self.assertIn("posterior mean", summary["reconstruction_method"])
        for role, path in zip(("baseline", "experiment"), self.paths):
            self.assertEqual(summary[role]["checkpoint_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(summary, json.loads((self.output / "summary.json").read_text()))
        for name in ("reconstruction_comparison.png", "generation_comparison.png", "validation_curves.png"):
            with Image.open(self.output / name) as image:
                self.assertGreater(image.width, 300)
                self.assertGreater(image.height, 200)

    def test_rejects_different_epochs_split_or_training_before_export(self):
        mutations = [
            ("epoch", lambda cp: cp.update(epoch=1)),
            ("fingerprint", lambda cp: cp.update(split_fingerprint="changed")),
            ("config", lambda cp: cp["config"]["training"].update(learning_rate=0.01)),
            ("config", lambda cp: cp["config"]["data"].update(image_size=32)),
            ("config", lambda cp: cp["config"]["model"].update(latent_dim=3)),
            ("config", lambda cp: cp["config"]["training"].update(reconstruction_scale=0.25)),
            ("config", lambda cp: cp["config"].update(seed=43)),
            ("config", lambda cp: cp["config"]["generation"].update(seed=124)),
        ]
        original = copy.deepcopy(self.experiment)
        for message, change in mutations:
            with self.subTest(message=message):
                self.experiment = copy.deepcopy(original)
                change(self.experiment)
                with self.assertRaisesRegex(ValueError, message):
                    self.run_comparison()
                self.assertFalse(self.output.exists())

    def test_128_comparison_labels_actual_resolution_and_scaled_metric_provenance(self):
        from matplotlib.figure import Figure

        for checkpoint in (self.baseline, self.experiment):
            checkpoint["config"]["data"]["image_size"] = 128
            checkpoint["config"]["training"]["reconstruction_scale"] = 0.25
        saved_titles = {}
        real_savefig = Figure.savefig
        def record_titles(figure, destination, *args, **kwargs):
            saved_titles[Path(destination).name] = [axis.get_title() for axis in figure.axes]
            return real_savefig(figure, destination, *args, **kwargs)
        with patch.object(Figure, "savefig", record_titles):
            summary = self.run_comparison()
        self.assertEqual(saved_titles["reconstruction_comparison.png"][0], "Real (128×128)")
        for role in ("baseline", "experiment"):
            self.assertEqual(summary[role]["image_size"], 128)
            self.assertEqual(summary[role]["reconstruction_scale"], 0.25)
        self.assertIn("scaled", summary["validation_metric_method"])

    def test_rejects_actual_changed_membership_even_if_checkpoint_fingerprints_match(self):
        Image.new("L", (10, 12), 100).save(self.data_root / "val/NORMAL/extra.png")
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            self.run_comparison()
        self.assertFalse(self.output.exists())

    def test_short_class_uses_available_examples_only(self):
        (self.data_root / "val/NORMAL/1.png").unlink()
        (self.data_root / "val/NORMAL/2.png").unlink()
        manifest = save_split_manifest(build_dataloaders(self.baseline["config"]), self.baseline["config"], self.root / "manifest.json")
        for checkpoint in (self.baseline, self.experiment):
            checkpoint["split_fingerprint"] = manifest["fingerprint"]
        summary = self.run_comparison()
        self.assertEqual(summary["reconstruction_source_files"], ["val/NORMAL/0.png", "val/PNEUMONIA/0.png", "val/PNEUMONIA/1.png"])

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

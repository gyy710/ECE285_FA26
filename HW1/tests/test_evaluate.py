"""Synthetic protocol checks; never construct/download the pretrained backbone."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn
from torchvision import transforms

import evaluate
from dataset import build_dataloaders, save_split_manifest


class PreprocessingTests(unittest.TestCase):
    def test_quantizes_before_bilinear_resize_and_normalizes_rgb(self):
        # Omitting quantization, using a center crop, or double-normalizing breaks this.
        images = torch.tensor([[[[0.0, 0.501], [1.0, 0.25]]]], dtype=torch.float32)
        actual = evaluate.preprocess_images(images)
        pixels = np.array([[0, 127], [255, 63]], dtype=np.uint8)
        image = Image.fromarray(pixels).resize((299, 299), Image.Resampling.BILINEAR).convert("RGB")
        expected = transforms.Normalize((.485, .456, .406), (.229, .224, .225))(
            transforms.ToTensor()(image)
        ).unsqueeze(0)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(tuple(actual.shape), (1, 3, 299, 299))

    def test_invalid_images_fail_instead_of_being_rescaled(self):
        examples = [torch.zeros(0, 1, 2, 2), torch.zeros(2, 3, 2, 2),
                    torch.zeros(2, 1, 2), torch.zeros(1, 1, 2, 2, dtype=torch.uint8),
                    torch.full((1, 1, 2, 2), float("nan")),
                    torch.full((1, 1, 2, 2), 1.01), torch.full((1, 1, 2, 2), -.01)]
        for images in examples:
            with self.subTest(shape=images.shape, dtype=images.dtype):
                with self.assertRaises(ValueError):
                    evaluate.preprocess_images(images)


class MetricTests(unittest.TestCase):
    def test_identical_constant_distributions_have_zero_fid_and_unit_is(self):
        probabilities = torch.tensor([[.2, .3, .5]]).repeat(5, 1)
        result = evaluate.compute_probability_metrics(probabilities, probabilities, 2)
        self.assertAlmostEqual(result["fid"], 0, delta=1e-10)
        self.assertAlmostEqual(result["inception_score"], 1, places=7)

    def test_confident_diversity_uses_global_score_and_preserves_short_last_batch(self):
        probabilities = torch.tensor([[1., 0, 0], [0, 1., 0], [1., 0, 0], [0, 0, 1.], [0, 1., 0]])
        expected_is = np.exp(-2 * .4 * np.log(.4) - .2 * np.log(.2))
        full = evaluate.compute_probability_metrics(probabilities, probabilities, 5)
        for batch in (1, 2, 3):
            result = evaluate.compute_probability_metrics(probabilities, probabilities, batch)
            self.assertAlmostEqual(result["inception_score"], expected_is, places=6)
            self.assertAlmostEqual(result["inception_score"], full["inception_score"], places=7)
            self.assertAlmostEqual(result["fid"], full["fid"], delta=1e-9)

    def test_nonidentical_feature_fid_matches_hand_computed_two_point_distance(self):
        real = torch.tensor([[1., 0, 0], [0, 1., 0]])
        fake = torch.tensor([[0, 0, 1.], [0, 0, 1.]])
        # ||(.5,.5,0)-(0,0,1)||²=1.5; trace(real sample covariance)=1.
        result = evaluate.compute_probability_metrics(real, fake, 1)
        self.assertAlmostEqual(result["fid"], 2.5, places=7)
        self.assertAlmostEqual(result["inception_score"], 1, places=7)

    def test_rejects_invalid_unequal_or_insufficient_probabilities(self):
        valid = torch.tensor([[.2, .3, .5]]).repeat(5, 1)
        pairs = [(valid[:0], valid[:0]), (valid[:1], valid[:1]), (valid, valid[:4]),
                 (valid, torch.ones(5, 2)), (valid, valid.double()),
                 (valid, valid * 2), (valid, torch.full_like(valid, float("nan"))),
                 (valid, torch.tensor([[-.1, .4, .7]]).repeat(5, 1))]
        for real, fake in pairs:
            with self.subTest(real=real.shape, fake=fake.shape):
                with self.assertRaises(ValueError):
                    evaluate.compute_probability_metrics(real, fake, 2)
        with self.assertRaises(ValueError):
            evaluate.compute_probability_metrics(valid, valid, 0)

    def test_records_numerical_warnings_and_propagates_sqrtm_failure(self):
        import warnings
        valid = torch.tensor([[.2, .3, .5]]).repeat(5, 1)
        def warned_compute(_):
            warnings.warn("singular synthetic covariance", RuntimeWarning)
            return .125
        with patch("evaluate.FID.compute", warned_compute):
            result = evaluate.compute_probability_metrics(valid, valid, 2)
        self.assertTrue(any("singular synthetic covariance" in item for item in result["numerical_warnings"]))
        with patch("evaluate.FID.compute", side_effect=ValueError("Imaginary component 1.0")):
            with self.assertRaisesRegex(RuntimeError, "CPU.*FID") as raised:
                evaluate.compute_probability_metrics(valid, valid, 2)
        self.assertIsInstance(raised.exception.__cause__, ValueError)
        with patch("evaluate.FID.compute", return_value=float("nan")):
            with self.assertRaisesRegex(RuntimeError, "finite"):
                evaluate.compute_probability_metrics(valid, valid, 2)


class TinyVAE(nn.Module):
    latent_dim = 2

    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.1))

    def decode(self, z):
        return (z[:, :1] + self.bias).reshape(-1, 1, 1, 1).expand(-1, 1, 128, 128)


class TinyBackbone(nn.Module):
    def forward(self, images):
        if self.training or torch.is_grad_enabled():
            raise AssertionError("feature extraction must use eval/no_grad")
        values = images.mean((1, 2, 3)).sigmoid()
        return torch.stack((values * .5, (1 - values) * .5, torch.full_like(values, .5)), dim=1)


class ExtractionTests(unittest.TestCase):
    def test_caches_original_float32_probabilities_without_second_softmax(self):
        images = torch.zeros(5, 1, 2, 2)
        batches = [(images[:2], torch.zeros(2)), (images[2:], torch.zeros(3))]
        extractor = TinyBackbone().eval()
        actual = evaluate.collect_probabilities(batches, extractor, "cpu", 5, 3, "real")
        with torch.no_grad():
            expected = extractor(evaluate.preprocess_images(images))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(actual.dtype, torch.float32)
        self.assertEqual(actual.device.type, "cpu")
        with self.assertRaisesRegex(ValueError, "count"):
            evaluate.collect_probabilities(batches, extractor, "cpu", 6, 3, "real")
        with self.assertRaises(ValueError):
            evaluate.collect_probabilities(batches, extractor, "cpu", 5, 1000, "real")


class EvaluationIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        data_root = self.root / "data"
        for split in ("train", "val", "test"):
            for label in ("NORMAL", "PNEUMONIA"):
                directory = data_root / split / label
                directory.mkdir(parents=True)
                count = 3 if split == "test" and label == "NORMAL" else 2
                for index in range(count):
                    mode = "RGB" if index % 2 else "L"
                    color = (40 + index * 30,) * 3 if mode == "RGB" else 40 + index * 30
                    Image.new(mode, (11, 9), color).save(directory / f"{index}.png")
        self.config = {
            "seed": 42, "data": {"root": str(data_root), "image_size": 128, "channels": 1, "num_workers": 0},
            "training": {"batch_size": 4}, "generation": {"seed": 123},
            "evaluation": {"real_split": "test", "sample_count": "match_real_split", "seed": 456,
                           "batch_size": 2, "device": "cpu", "implementation": "ignite_default", "inception_input_size": 299}}
        self.manifest = save_split_manifest(build_dataloaders(self.config), self.config, self.root / "source.json")
        self.checkpoint = self.root / "best.pt"
        self.payload = {"epoch": 30, "config": self.config, "split_fingerprint": self.manifest["fingerprint"]}
        torch.save(self.payload, self.checkpoint)
        self.output = self.root / "evaluation"
        self.args = ["--checkpoint", str(self.checkpoint), "--output-dir", str(self.output)]
        self.addCleanup(patch.stopall)
        patch("evaluate.FEATURE_DIMENSION", 3).start()
        patch("evaluate.create_extractor", return_value=TinyBackbone().eval()).start()
        patch("evaluate.load_trained_model", side_effect=lambda path, device: (
            TinyVAE().to(device).eval(), torch.load(path, map_location="cpu", weights_only=True))).start()

    def test_all_test_images_and_fresh_prior_are_auditable_and_overwrite_is_refused(self):
        summary = evaluate.main(self.args)
        saved = json.loads((self.output / "metrics.json").read_text())
        self.assertEqual(summary, saved)
        self.assertEqual(saved["num_real_samples"], 5)
        self.assertEqual(saved["num_generated_samples"], 5)
        self.assertEqual(saved["seed"], 456)
        self.assertEqual(saved["checkpoint_epoch"], 30)
        self.assertEqual(saved["checkpoint_sha256"], hashlib.sha256(self.checkpoint.read_bytes()).hexdigest())
        self.assertEqual(saved["split_fingerprint"], self.manifest["fingerprint"])
        paths = ["test/NORMAL/0.png", "test/NORMAL/1.png", "test/NORMAL/2.png", "test/PNEUMONIA/0.png", "test/PNEUMONIA/1.png"]
        self.assertEqual(saved["real_paths"], paths)
        self.assertEqual(saved["test_class_counts"], {"NORMAL": 3, "PNEUMONIA": 2})
        self.assertEqual(saved["metric_device"], "cpu")
        self.assertEqual(saved["feature_device"], "cpu")
        self.assertEqual(saved["generator_device"], "cpu")
        self.assertEqual(saved["feature_dimension"], 3)
        self.assertEqual(saved["feature_implementation"], "ignite_default_1000_probabilities")
        for name in ("python", "torch", "torchvision", "ignite", "numpy", "scipy", "Pillow"):
            self.assertTrue(saved["versions"][name])
        self.assertTrue(saved["preprocessing"]["internal_transform_input"])
        self.assertTrue(saved["limitations"])
        self.assertGreater(saved["elapsed_seconds"], 0)
        cache = torch.load(self.output / "feature_probabilities.pt", weights_only=True)
        self.assertEqual(cache["real_paths"], paths)
        self.assertEqual(cache["seed"], 456)
        self.assertEqual(cache["checkpoint_sha256"], saved["checkpoint_sha256"])
        self.assertEqual(tuple(cache["real"].shape), (5, 3))
        self.assertEqual(tuple(cache["generated"].shape), (5, 3))
        self.assertEqual(cache["generated"].dtype, torch.float32)
        # Independently derive expected pixels/probabilities: catches a wrong seed,
        # missing truncation, or mismatched cached real path order.
        def probabilities_for_pixels(pixels):
            mean = torch.tensor([.485, .456, .406])
            std = torch.tensor([.229, .224, .225])
            values = ((pixels[:, None] / 255 - mean) / std).mean(1).sigmoid()
            return torch.stack((values * .5, (1 - values) * .5, torch.full_like(values, .5)), 1)
        expected_real = probabilities_for_pixels(torch.tensor([40., 70., 100., 40., 70.]))
        latent = torch.randn(5, 2, generator=torch.Generator().manual_seed(456))
        fake_pixels = ((latent[:, 0] + .1).sigmoid() * 255).to(torch.uint8).float()
        expected_generated = probabilities_for_pixels(fake_pixels)
        torch.testing.assert_close(cache["real"], expected_real)
        torch.testing.assert_close(cache["generated"], expected_generated)
        with Image.open(self.output / "evaluation_generated_grid.png") as image:
            self.assertGreater(image.width, 128)
        self.assertEqual(json.loads((self.output / "evaluation_manifest.json").read_text()), self.manifest)
        recomputed = evaluate.compute_probability_metrics(cache["real"], cache["generated"], 3)
        self.assertAlmostEqual(recomputed["fid"], saved["fid"], places=8)
        self.assertAlmostEqual(recomputed["inception_score"], saved["inception_score"], places=8)
        with self.assertRaises(FileExistsError):
            evaluate.main(self.args)

    def test_split_mismatch_fails_before_exports(self):
        self.payload["split_fingerprint"] = "changed"
        torch.save(self.payload, self.checkpoint)
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            evaluate.main(self.args)
        self.assertFalse(self.output.exists())

    def test_incompatible_evaluation_config_fails_before_exports(self):
        for key, value in (("real_split", "val"), ("sample_count", 64), ("implementation", "pool3"),
                           ("inception_input_size", 342), ("batch_size", 0), ("device", "cuda")):
            payload = copy.deepcopy(self.payload)
            payload["config"]["evaluation"][key] = value
            torch.save(payload, self.checkpoint)
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    evaluate.main(self.args)
                self.assertFalse(self.output.exists())

    def test_metric_failure_leaves_no_partial_exports(self):
        with patch("evaluate.compute_probability_metrics", side_effect=RuntimeError("invalid sqrtm")):
            with self.assertRaisesRegex(RuntimeError, "sqrtm"):
                evaluate.main(self.args)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()

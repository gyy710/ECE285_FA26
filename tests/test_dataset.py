"""Data-pipeline checks for split isolation and mixed source image formats."""

from pathlib import Path
import tempfile
import unittest

from PIL import Image
import torch

from dataset import build_dataloaders


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for split in ("train", "val", "test"):
            for label in ("NORMAL", "PNEUMONIA"):
                (self.root / split / label).mkdir(parents=True)
            Image.new("L", (31, 19), 0).save(self.root / split / "NORMAL/gray.png")
            Image.new("RGB", (17, 29), (255, 255, 255)).save(
                self.root / split / "PNEUMONIA/rgb.png"
            )
            Image.new("L", (23, 13), 128).save(
                self.root / split / "PNEUMONIA/mid.png"
            )
            # AppleDouble files can have image extensions but are not images.
            (self.root / split / "NORMAL/._gray.png").write_bytes(b"metadata")
        duplicate = self.root / "chest_xray/train/NORMAL"
        duplicate.mkdir(parents=True)
        Image.new("L", (8, 8)).save(duplicate / "extra.png")
        (self.root / "__MACOSX").mkdir()
        self.config = {
            "seed": 42,
            "data": {"root": str(self.root), "image_size": 64, "channels": 1, "num_workers": 0},
            "training": {"batch_size": 2},
        }

    def test_mixed_images_are_loaded_once_and_preprocessed_consistently(self):
        loaders = build_dataloaders(self.config)
        for split, loader in loaders.items():
            self.assertEqual(len(loader.dataset), 3, split)
            batches = list(loader)
            # The last sample must not be lost, especially during evaluation.
            self.assertEqual([len(images) for images, _ in batches], [2, 1])
            images = torch.cat([images for images, _ in batches])
            self.assertEqual(tuple(images.shape), (3, 1, 64, 64))
            self.assertEqual(images.dtype, torch.float32)
            self.assertTrue(torch.isfinite(images).all().item())
            self.assertEqual(images.min().item(), 0.0)
            self.assertEqual(images.max().item(), 1.0)
        # Test labels follow the source class, with no random reordering.
        test_labels = torch.cat([labels for _, labels in loaders["test"]])
        self.assertEqual(test_labels.tolist(), [0, 1, 1])

    def test_seed_reproduces_initial_training_order(self):
        first = list(build_dataloaders(self.config)["train"])
        second = list(build_dataloaders(self.config)["train"])
        for (images_a, labels_a), (images_b, labels_b) in zip(first, second):
            self.assertTrue(torch.equal(images_a, images_b))
            self.assertTrue(torch.equal(labels_a, labels_b))

    def test_missing_split_has_an_actionable_error(self):
        (self.root / "test").rename(self.root / "held_out")
        with self.assertRaisesRegex(FileNotFoundError, "test"):
            build_dataloaders(self.config)


class GroupedValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for split in ("train", "val", "test"):
            for label in ("NORMAL", "PNEUMONIA"):
                (self.root / split / label).mkdir(parents=True)
        for group in range(1, 5):
            for image in (1, 2):
                Image.new("L", (12, 10), 80).save(
                    self.root / "train/NORMAL" / f"IM-{group:04d}-{image:04d}.jpeg"
                )
                infection = "bacteria" if image == 1 else "virus"
                Image.new("L", (12, 10), 160).save(
                    self.root / "train/PNEUMONIA" / f"person{group}_{infection}_1.jpeg"
                )
        for split, number in (("val", 100), ("test", 200)):
            Image.new("L", (12, 10), 80).save(
                self.root / split / "NORMAL" / f"IM-{number:04d}-0001.jpeg"
            )
            Image.new("L", (12, 10), 160).save(
                self.root / split / "PNEUMONIA" / f"person{number}_bacteria_1.jpeg"
            )
        self.config = {
            "seed": 42,
            "data": {
                "root": str(self.root), "image_size": 64, "channels": 1,
                "num_workers": 0, "validation_fraction": 0.25,
            },
            "training": {"batch_size": 2},
        }

    def test_holdout_keeps_candidate_groups_together_and_preserves_test(self):
        from dataset import filename_group_key

        loaders = build_dataloaders(self.config)
        self.assertEqual({k: len(v.dataset) for k, v in loaders.items()}, {"train": 12, "val": 4, "test": 2})
        samples = {split: set(path for path, _ in loader.dataset.samples) for split, loader in loaders.items()}
        self.assertFalse(samples["train"] & samples["val"])
        original_train = {str(p) for p in (self.root / "train").rglob("*.jpeg")}
        self.assertEqual(samples["train"] | samples["val"], original_train)
        self.assertTrue(all(Path(p).is_relative_to(self.root / "test") for p in samples["test"]))
        groups = {split: {filename_group_key(path) for path in paths} for split, paths in samples.items()}
        self.assertFalse(groups["train"] & groups["val"])
        for split in ("train", "val"):
            self.assertEqual(set(loaders[split].dataset.targets), {0, 1})
        repeated = build_dataloaders(self.config)
        self.assertEqual(loaders["val"].dataset.samples, repeated["val"].dataset.samples)

    def test_group_keys_preserve_normal_namespaces_and_merge_infection_suffixes(self):
        from dataset import filename_group_key

        self.assertNotEqual(
            filename_group_key("NORMAL/IM-0001-0001.jpeg"),
            filename_group_key("NORMAL/NORMAL2-IM-0001-0001.jpeg"),
        )
        self.assertEqual(
            filename_group_key("PNEUMONIA/person5_bacteria_1.jpeg"),
            filename_group_key("PNEUMONIA/person5_virus_2.jpeg"),
        )

    def test_manifest_records_actual_sources_and_reproducible_fingerprint(self):
        import json
        from dataset import save_split_manifest

        loaders = build_dataloaders(self.config)
        destination = self.root / "outputs/split_manifest.json"
        manifest = save_split_manifest(loaders, self.config, destination)
        self.assertEqual(json.loads(destination.read_text()), manifest)
        self.assertEqual(manifest["counts"], {"train": 12, "val": 4, "test": 2})
        self.assertEqual(len(manifest["unused_original_val"]), 2)
        self.assertTrue(all(entry["path"].startswith("train/") for entry in manifest["splits"]["val"]))
        second = save_split_manifest(build_dataloaders(self.config), self.config, destination)
        self.assertEqual(manifest["fingerprint"], second["fingerprint"])

    def test_invalid_validation_fraction_fails_before_training(self):
        self.config["data"]["validation_fraction"] = 1.0
        with self.assertRaisesRegex(ValueError, "validation_fraction"):
            build_dataloaders(self.config)


if __name__ == "__main__":
    unittest.main()

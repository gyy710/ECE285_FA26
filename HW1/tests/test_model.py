"""Checks for VAE tensor shapes, loss scaling and gradient flow."""

import math
import unittest

import torch

from model import ConvVAE, build_model, vae_loss


def small_config():
    return {
        "data": {"image_size": 64, "channels": 1},
        "model": {"latent_dim": 8},
    }


class ModelTests(unittest.TestCase):
    def test_shapes_include_a_single_image_batch_and_prior_samples(self):
        model = build_model(small_config())
        with torch.no_grad():
            for batch_size in (1, 3):
                images = torch.rand(batch_size, 1, 64, 64)
                logits, mu, logvar = model(images)
                self.assertEqual(logits.shape, images.shape)
                self.assertEqual(tuple(mu.shape), (batch_size, 8))
                self.assertEqual(logvar.shape, mu.shape)
                self.assertTrue(torch.isfinite(logits).all().item())
            generated = model.decode(torch.randn(5, 8)).sigmoid()
        self.assertEqual(tuple(generated.shape), (5, 1, 64, 64))
        self.assertGreaterEqual(generated.min().item(), 0.0)
        self.assertLessEqual(generated.max().item(), 1.0)

    def test_reparameterization_matches_noise_and_keeps_both_gradient_paths(self):
        model = build_model(small_config())
        mu = torch.tensor([[1.0, -2.0]], requires_grad=True)
        logvar = torch.tensor([[math.log(4.0), math.log(0.25)]], requires_grad=True)
        torch.manual_seed(17)
        noise = torch.randn_like(mu)
        torch.manual_seed(17)
        z = model.reparameterize(mu, logvar)
        expected_std = torch.tensor([[2.0, 0.5]])
        torch.testing.assert_close(z, mu + expected_std * noise)
        z.sum().backward()
        torch.testing.assert_close(mu.grad, torch.ones_like(mu))
        torch.testing.assert_close(logvar.grad, 0.5 * expected_std * noise)

    def test_known_loss_values_and_beta_weight(self):
        # Check the analytic formula in float64 to avoid rounding across 4096 float32 terms.
        images = torch.rand(2, 1, 64, 64, dtype=torch.float64)
        logits = torch.zeros_like(images)
        mu = torch.zeros(2, 8, dtype=images.dtype)
        logvar = torch.zeros_like(mu)
        total, reconstruction, kl = vae_loss(logits, images, mu, logvar)
        self.assertAlmostEqual(reconstruction.item(), 4096 * math.log(2), places=8)
        self.assertEqual(kl.item(), 0.0)
        torch.testing.assert_close(total, reconstruction)
        # For unit variance, KL(N(ones, I) || N(zeros, I)) = 8 / 2.
        total, reconstruction, kl = vae_loss(logits, images, torch.ones_like(mu), logvar, beta=0.25)
        self.assertEqual(kl.item(), 4.0)
        torch.testing.assert_close(total, reconstruction + 1.0)
        # With zero mean and variance 4, each dimension contributes (4 - 1 - log(4)) / 2 to KL.
        _, _, kl = vae_loss(logits, images, mu, torch.full_like(mu, math.log(4)))
        self.assertAlmostEqual(kl.item(), 4 * (3 - math.log(4)), places=8)

    def test_eval_keeps_sampling_but_decoding_the_mean_is_deterministic(self):
        torch.manual_seed(23)
        model = build_model(small_config()).eval()
        images = torch.rand(1, 1, 64, 64)
        with torch.no_grad():
            logits_a, mu_a, _ = model(images)
            logits_b, mu_b, _ = model(images)
            mean_reconstruction_a = model.decode(mu_a)
            mean_reconstruction_b = model.decode(mu_b)
        torch.testing.assert_close(mu_a, mu_b)
        self.assertFalse(torch.equal(logits_a, logits_b))
        torch.testing.assert_close(mean_reconstruction_a, mean_reconstruction_b)

    def test_duplicating_the_batch_does_not_scale_the_loss(self):
        images = torch.rand(2, 1, 64, 64)
        logits = torch.randn_like(images)
        mu, logvar = torch.randn(2, 8), torch.randn(2, 8)
        original = vae_loss(logits, images, mu, logvar)
        repeated = vae_loss(
            logits.repeat(3, 1, 1, 1), images.repeat(3, 1, 1, 1),
            mu.repeat(3, 1), logvar.repeat(3, 1),
        )
        for before, after in zip(original, repeated):
            torch.testing.assert_close(before, after)

    def test_extreme_logits_have_finite_loss_and_gradients(self):
        logits = torch.tensor([[[[1000.0, -1000.0]]]], requires_grad=True)
        images = torch.tensor([[[[0.0, 1.0]]]])
        mu = torch.zeros(1, 2, requires_grad=True)
        logvar = torch.zeros_like(mu, requires_grad=True)
        total, _, _ = vae_loss(logits, images, mu, logvar)
        self.assertEqual(total.item(), 2000.0)
        total.backward()
        self.assertTrue(torch.isfinite(logits.grad).all().item())

    def test_end_to_end_backward_reaches_all_model_parameters(self):
        torch.manual_seed(42)
        model = build_model(small_config())
        images = torch.rand(2, 1, 64, 64)
        logits, mu, logvar = model(images)
        total, _, _ = vae_loss(logits, images, mu, logvar)
        total.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all().item(), name)
            self.assertGreater(parameter.grad.abs().sum().item(), 0.0, name)

    def test_unsupported_image_resolution_is_reported_early(self):
        config = small_config()
        config["data"]["image_size"] = 96
        with self.assertRaisesRegex(ValueError, "64.*128"):
            build_model(config)

    def test_128_shapes_prior_samples_and_finite_backward(self):
        config = small_config()
        config["data"]["image_size"] = 128
        model = build_model(config)
        self.assertEqual(model.image_size, 128)
        for batch_size in (1, 3):
            with self.subTest(batch_size=batch_size):
                model.zero_grad(set_to_none=True)
                images = torch.rand(batch_size, 1, 128, 128)
                logits, mu, logvar = model(images)
                self.assertEqual(logits.shape, images.shape)
                self.assertEqual(tuple(mu.shape), (batch_size, 8))
                self.assertEqual(logvar.shape, mu.shape)
                for tensor in (logits, mu, logvar):
                    self.assertTrue(torch.isfinite(tensor).all().item())
                total, _, _ = vae_loss(logits, images, mu, logvar, reconstruction_scale=0.25)
                self.assertTrue(torch.isfinite(total).item())
                total.backward()
                for name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all().item(), name)
        with torch.no_grad():
            generated = model.decode(torch.randn(5, 8)).sigmoid()
        self.assertEqual(tuple(generated.shape), (5, 1, 128, 128))
        self.assertTrue(torch.isfinite(generated).all().item())
        self.assertGreaterEqual(generated.min().item(), 0.0)
        self.assertLessEqual(generated.max().item(), 1.0)

    def test_default_64_model_accepts_the_legacy_state_dict_strictly(self):
        # These names/shapes are the saved baseline checkpoint contract.
        expected_shapes = {
            "encoder.0.weight": (32, 1, 4, 4), "encoder.0.bias": (32,),
            "encoder.2.weight": (64, 32, 4, 4), "encoder.2.bias": (64,),
            "encoder.4.weight": (128, 64, 4, 4), "encoder.4.bias": (128,),
            "encoder.6.weight": (256, 128, 4, 4), "encoder.6.bias": (256,),
            "fc_mu.weight": (128, 4096), "fc_mu.bias": (128,),
            "fc_logvar.weight": (128, 4096), "fc_logvar.bias": (128,),
            "decoder_input.weight": (4096, 128), "decoder_input.bias": (4096,),
            "decoder.0.weight": (256, 128, 4, 4), "decoder.0.bias": (128,),
            "decoder.2.weight": (128, 64, 4, 4), "decoder.2.bias": (64,),
            "decoder.4.weight": (64, 32, 4, 4), "decoder.4.bias": (32,),
            "decoder.6.weight": (32, 1, 4, 4), "decoder.6.bias": (1,),
        }
        model = ConvVAE()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 2955201)
        self.assertEqual({name: tuple(value.shape) for name, value in model.state_dict().items()}, expected_shapes)
        legacy_state = {name: torch.zeros(shape) for name, shape in expected_shapes.items()}
        result = model.load_state_dict(legacy_state, strict=True)
        self.assertEqual(result.missing_keys, [])
        self.assertEqual(result.unexpected_keys, [])
        with torch.no_grad():
            logits = model.decode(torch.zeros(1, 128))
        self.assertEqual(tuple(logits.shape), (1, 1, 64, 64))
        self.assertTrue(torch.equal(logits, torch.zeros_like(logits)))

    def test_constructor_and_encoder_reject_unsupported_input_shapes(self):
        with self.assertRaisesRegex(ValueError, "64.*128"):
            ConvVAE(image_size=96)
        for image_size, wrong_size in ((64, 128), (128, 64)):
            model = ConvVAE(latent_dim=8, image_size=image_size)
            with self.assertRaisesRegex(ValueError, str(image_size)):
                model.encode(torch.rand(1, 1, wrong_size, wrong_size))
            with self.assertRaises(ValueError):
                model.encode(torch.rand(1, 3, image_size, image_size))

    def test_build_model_rejects_multiple_channels_at_both_resolutions(self):
        for image_size in (64, 128):
            config = small_config()
            config["data"].update(image_size=image_size, channels=3)
            with self.assertRaises(ValueError):
                build_model(config)

    def test_scaled_128_reconstruction_matches_64_without_scaling_kl(self):
        mu = torch.ones(2, 8, dtype=torch.float64)
        logvar = torch.zeros_like(mu)
        results = []
        for image_size, scale in ((64, 1.0), (128, 0.25)):
            images = torch.rand(2, 1, image_size, image_size, dtype=torch.float64)
            results.append(vae_loss(
                torch.zeros_like(images), images, mu, logvar,
                beta=0.5, reconstruction_scale=scale,
            ))
        for result in results:
            total, reconstruction, kl = result
            self.assertAlmostEqual(reconstruction.item(), 4096 * math.log(2), places=8)
            self.assertEqual(kl.item(), 4.0)
            torch.testing.assert_close(total, reconstruction + 2.0)
        for baseline, scaled in zip(*results):
            torch.testing.assert_close(baseline, scaled)

    def test_reconstruction_scale_must_be_finite_and_positive(self):
        images = torch.zeros(1, 1, 2, 2)
        mu = torch.zeros(1, 2)
        for scale in (0.0, -1.0, math.inf, -math.inf, math.nan):
            with self.subTest(scale=scale), self.assertRaisesRegex(ValueError, "reconstruction_scale"):
                vae_loss(images, images, mu, mu, reconstruction_scale=scale)


if __name__ == "__main__":
    unittest.main()

"""IS / FID evaluation following https://pytorch-ignite.ai/blog/gan-evaluation-with-fid-and-is/.

Differences from the blog, both kept deliberately:
* images are mapped from [-1, 1] back to [0, 1] before ``ToPILImage`` (the blog
  feeds tanh outputs directly, which wraps negative values);
* besides ignite's default FID (computed on the 1000-d Inception class
  probabilities, which is what the blog reports), we also report the standard
  FID on 2048-d pool3 features.
"""

import PIL.Image as Image
import torch
import torchvision.transforms as transforms
from ignite.engine import Engine
from ignite.metrics import FID, InceptionScore
from ignite.metrics.gan.utils import InceptionModel
from ignite.metrics.metric import reinit__is_reduced


class _DeviceInception(torch.nn.Module):
    """Runs Inception on ``device`` but returns CPU features.

    ignite moves ``feature_extractor`` to the metric device; keeping the network
    inside a list hides it from ``.to()`` so the heavy forward pass stays on the
    accelerator while the covariance statistics are accumulated in float64 on CPU
    (MPS has no float64). The last result is cached because the same fake batch
    is fed to both FID and IS.
    """

    def __init__(self, return_features, device):
        super().__init__()
        self._net = [InceptionModel(return_features=return_features, device=device)]
        self._cache = []

    def forward(self, x):
        key = (x.data_ptr(), tuple(x.shape))
        for k, src, out in self._cache:
            if k == key and src is not None:
                return out
        out = self._net[0](x).float().cpu()
        # keep a reference to the source tensor so its storage (and key) stays unique
        self._cache = (self._cache + [(key, x, out)])[-2:]
        return out


class BatchedFID(FID):
    """ignite FID with the per-sample outer-product loop replaced by F^T F."""

    @reinit__is_reduced
    def update(self, output):
        train, test = output
        train_features = self._extract_features(train)
        test_features = self._extract_features(test)
        self._train_total += train_features.sum(0)
        self._train_sigma += train_features.T @ train_features
        self._test_total += test_features.sum(0)
        self._test_sigma += test_features.T @ test_features
        self._num_examples += train_features.shape[0]


def interpolate(batch):
    """[-1, 1] tensors (B,3,H,W) -> [0, 1] tensors (B,3,299,299) via PIL bilinear."""
    arr = []
    for img in batch:
        pil_img = transforms.ToPILImage()((img.detach().cpu().float() + 1) / 2)
        resized_img = pil_img.resize((299, 299), Image.BILINEAR)
        arr.append(transforms.ToTensor()(resized_img))
    return torch.stack(arr)


class GANEvaluator:
    def __init__(self, device):
        self.device = device
        probs = _DeviceInception(return_features=False, device=device)
        pool = _DeviceInception(return_features=True, device=device)
        self.fid_probs = BatchedFID(num_features=1000, feature_extractor=probs, device="cpu")
        self.fid_pool = BatchedFID(num_features=2048, feature_extractor=pool, device="cpu")
        self.is_metric = InceptionScore(
            num_features=1000, feature_extractor=probs, device="cpu", output_transform=lambda x: x[0]
        )
        self.is_real = InceptionScore(
            num_features=1000, feature_extractor=probs, device="cpu", output_transform=lambda x: x[1]
        )

    @torch.no_grad()
    def evaluate(self, netG, real_images, latent_dim, batch_size=200, real_is=False):
        """real_images: uint8 tensor (N,3,64,64). Generates N fakes and compares."""
        was_training = netG.training
        netG.eval()
        n = real_images.shape[0]
        batches = [real_images[i : i + batch_size] for i in range(0, n, batch_size)]

        def evaluation_step(engine, batch):
            noise = torch.randn(batch.shape[0], latent_dim, 1, 1, device=self.device)
            fake = interpolate(netG(noise))
            real = interpolate(batch.float() / 127.5 - 1)
            return fake, real

        evaluator = Engine(evaluation_step)
        self.fid_probs.attach(evaluator, "fid")
        self.fid_pool.attach(evaluator, "fid2048")
        self.is_metric.attach(evaluator, "is")
        if real_is:
            self.is_real.attach(evaluator, "is_real")
        evaluator.run(batches, max_epochs=1)
        netG.train(was_training)
        return dict(evaluator.state.metrics)

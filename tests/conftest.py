import torch
import pytest


@pytest.fixture
def target_synth():
    """A pair of correlated (not identical) image batches: `synth` is
    `target` plus small noise, so metrics computed on them should be
    small but non-zero, and identical-input tests can compare against
    (target, target)."""
    torch.manual_seed(0)
    target = torch.randn(4, 3, 16, 16)
    synth = target + 0.05 * torch.randn(4, 3, 16, 16)
    return target, synth


class TinyCNN(torch.nn.Module):
    """Minimal stand-in for texture_metrics.criteria.cnn.CNN: returns
    a short list of feature maps without downloading any pretrained
    weights, for fast/offline tests of CNN-based metrics."""

    def __init__(self, layers_weights=(1.0, 1.0)):
        super().__init__()
        self.layers_weights = list(layers_weights)

    def forward(self, x: torch.Tensor):
        return [x, x.mean(dim=1, keepdim=True)]

    def compile(self):
        return self


@pytest.fixture
def tiny_cnn():
    return TinyCNN()


class DummyGenerativeModel(torch.nn.Module):
    """Stand-in for the `model` argument of `metrics_loop`: takes a
    batch of real images and returns {"sample": ..., "target": ...},
    matching the interface metrics_loop expects."""

    def __init__(self, noise_std: float = 0.01):
        super().__init__()
        self.noise_std = noise_std

    def forward(self, batch: torch.Tensor, **kwargs):
        target = batch
        sample = target + self.noise_std * torch.randn_like(target)
        return {"sample": sample, "target": target}


@pytest.fixture
def dummy_model():
    return DummyGenerativeModel()


class DummyLoader:
    """Minimal loader yielding a fixed number of image batches."""

    def __init__(self, n_batches: int = 3, batch_size: int = 4, size: int = 16):
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.size = size

    def __iter__(self):
        for _ in range(self.n_batches):
            yield torch.randn(self.batch_size, 3, self.size, self.size)

    def __len__(self):
        return self.n_batches


@pytest.fixture
def dummy_loader():
    return DummyLoader()

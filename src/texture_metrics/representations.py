from typing import Callable

import torch

from .criteria.fourier import radial_profile, spectral_slope, spectral_polynomial_embedding


_repr_dict: dict[str, Callable] = dict()


def register_representation(func: Callable):
    assert callable(func)
    _repr_dict[func.__name__] = func
    return func


register_representation(spectral_slope)
register_representation(spectral_polynomial_embedding)


@register_representation
def flatten(x: torch.Tensor) -> torch.Tensor:
    """Flattens all non-batch dimensions, turning a batch of images
    into a batch of feature vectors. Generic fallback representation
    for distance functions that expect (B, D) vectors."""
    return x.flatten(start_dim=1)


@register_representation
def color_mean(x: torch.Tensor) -> torch.Tensor:
    return x.mean(dim=(-2, -1))


@register_representation
def color_covariance(x: torch.Tensor) -> torch.Tensor:
    b, c, h, w = x.shape
    pixels = x.reshape(b, c, h * w)
    pixels = pixels - pixels.mean(dim=-1, keepdim=True)
    cov = torch.bmm(pixels, pixels.transpose(1, 2)) / (h * w - 1)
    idx = torch.triu_indices(c, c)
    return cov[:, idx[0], idx[1]]


@register_representation
def radial_spectrum(x: torch.Tensor, bin_size: int = 1) -> torch.Tensor:
    return radial_profile(x.mean(dim=-3), bin_size=bin_size)


@register_representation
def cnn_activations(
    x: torch.Tensor,
    cnn: torch.nn.Module,
    return_means: bool = True,
    return_stds: bool = True,
) -> dict[str, torch.Tensor]:
    feature_list = cnn(x)

    means, stds = [], []
    for features in feature_list:
        if return_means:
            mean = features.mean(dim=(-2, -1))
            means.append(mean)
        if return_stds:
            std = features.std(dim=(-2, -1))
            stds.append(std)

    activations = dict()
    if return_means:
        activations.update(
            dict(zip([f"mean_{i}" for i in range(len(means))], means))
        )
    if return_stds:
        activations.update(
            dict(zip([f"std_{i}" for i in range(len(stds))], stds))
        )
    return activations


@register_representation
def cnn_activations_summary(
    x: torch.Tensor,
    cnn: torch.nn.Module,
    summary: str = "gram",
) -> torch.Tensor:
    feature_list = cnn(x)

    if summary == "gram":
        summary_fn = lambda f: torch.bmm(
            f.flatten(-2), f.flatten(-2).transpose(-1, -2)
        )
    elif summary == "mean":
        summary_fn = lambda f: f.mean(dim=(-2, -1))
    elif summary == "std":
        summary_fn = lambda f: f.std(dim=(-2, -1))
    elif summary == "covariance":
        summary_fn = lambda f: torch.cov(f.flatten(-2).T)
    else:
        raise ValueError(f"Unknown summary: {summary}")

    return summary_fn

class RadialProfileExtractor(torch.nn.Module):
    def __init__(self, bin_size: int = 1):
        super().__init__()
        self.bin_size = bin_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return radial_profile(x.mean(dim=-3), bin_size=self.bin_size)


class LogRadialProfileExtractor(torch.nn.Module):
    def __init__(self, bin_size: int = 1):
        super().__init__()
        self.bin_size = bin_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return radial_profile(x.mean(dim=-3), bin_size=self.bin_size).log()

from typing import Callable

import torch

from .criteria import optimal_transport


_dist_dict: dict[str, Callable] = dict()


def register_distance(func: Callable):
    assert callable(func)
    _dist_dict[func.__name__] = func
    return func


def _matrix_sqrt_eigh(m: torch.Tensor) -> torch.Tensor:
    """Symmetric positive semi-definite matrix square root."""
    eigvals, eigvecs = torch.linalg.eigh(m)
    eigvals = eigvals.clamp(min=0)
    return eigvecs @ torch.diag(eigvals.sqrt()) @ eigvecs.T


@register_distance
def frechet_distance(real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    """Fréchet distance between two sets of feature vectors.

    Fits a Gaussian to each set and computes:
        d² = ||μ_r - μ_f||² + Tr(Σ_r + Σ_f - 2 (Σ_r Σ_f)^{1/2})

    Args:
        real (torch.Tensor): real features, shape (N, D).
        fake (torch.Tensor): synthetic features, shape (M, D).

    Returns:
        torch.Tensor: scalar Fréchet distance.
    """
    real, fake = real.double(), fake.double()
    mu_r, mu_f = real.mean(0), fake.mean(0)
    sigma_r = torch.cov(real.T)
    sigma_f = torch.cov(fake.T)

    diff = mu_r - mu_f
    sqrt_r = _matrix_sqrt_eigh(sigma_r)
    sqrt_product = _matrix_sqrt_eigh(sqrt_r @ sigma_f @ sqrt_r)

    fid = diff @ diff + torch.trace(sigma_r + sigma_f - 2 * sqrt_product)
    return fid.clamp(min=0).float()


@register_distance
def sliced_wasserstein_distance(
    real: torch.Tensor,
    fake: torch.Tensor,
    nslice: int = 1000,
    bslice: int = 100,
) -> torch.Tensor:
    """Sliced Wasserstein distance between two sets of feature vectors.

    Reshapes (N, D) feature vectors to (N, D, 1, 1) image-like tensors
    and delegates to ``texture_metrics.criteria.optimal_transport``.

    Args:
        real (torch.Tensor): real features, shape (N, D).
        fake (torch.Tensor): synthetic features, shape (M, D).
        nslice (int): number of slices for Sliced Wasserstein distance.
        bslice (int): batch size for Sliced Wasserstein distance.

    Returns:
        torch.Tensor: scalar SWD.
    """
    return optimal_transport.sliced_wasserstein_distance(
        real.unsqueeze(-1).unsqueeze(-1).transpose(0, -1),
        fake.unsqueeze(-1).unsqueeze(-1).transpose(0, -1),
        nslice=nslice,
        batch_size=bslice,
    )


@register_distance
def histogram_distance(real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    """L1 distance between histograms of feature vectors.

    Delegates to ``texture_metrics.criteria.optimal_transport``.

    Args:
        real (torch.Tensor): real features, shape (N, D).
        fake (torch.Tensor): synthetic features, shape (M, D).

    Returns:
        torch.Tensor: scalar histogram distance.
    """
    return optimal_transport.histogram_loss1D(real, fake)


@register_distance
def per_bin_wasserstein_distance(
    real: torch.Tensor, fake: torch.Tensor, p: int = 2
) -> dict[str, torch.Tensor]:
    """1D Wasserstein distance computed independently per coordinate.

    ``histogram_loss1D`` flattens its last two dims into one empirical
    distribution per leading batch element, so feeding it (D, N, 1)
    tensors (coordinates as the batch dim, samples as the distribution)
    yields one Wasserstein distance per coordinate instead of pooling
    all coordinates into a single distribution.

    Args:
        real (torch.Tensor): real features, shape (N, D).
        fake (torch.Tensor): synthetic features, shape (M, D).
        p (int): distance order.

    Returns:
        dict[str, torch.Tensor]: ``{"bin_<i>": distance}`` for each of
            the D coordinates.
    """
    real = real.T.unsqueeze(-1)
    fake = fake.T.unsqueeze(-1)
    dist = optimal_transport.histogram_loss1D(real, fake, p=p) ** (1 / p)
    return {f"bin_{i}": dist[i] for i in range(dist.shape[0])}

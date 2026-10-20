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
def monge_inception_distance(
    real: torch.Tensor,
    fake: torch.Tensor,
    rng_seed: int = 0,
    n_projections=100,
    projection_batch_size: int | None = None,
) -> torch.Tensor:
    """Monge Inception distance (MIND) between two sets of feature vectors.

    Squared sliced Wasserstein distance over ``n_projections`` random
    directions, scaled by ``3 * D`` so that its magnitude does not
    shrink with the feature dimension.

    Args:
        real (torch.Tensor): real features, shape (N, D).
        fake (torch.Tensor): synthetic features, shape (M, D).
        rng_seed (int): random seed for reproducibility.
        n_projections (int): number of projections to use.
        projection_batch_size (int, optional): number of projections
            processed at once, bounding the (N, batch) projected and
            sorted buffers. All directions are drawn upfront, so the
            result does not depend on it. None processes them all at once.

    Returns:
        torch.Tensor: scalar MIND.
    """
    n, d = real.shape
    assert n == fake.shape[0] and d == fake.shape[1], (
        f"real and fake must have the same shape, got {real.shape} and {fake.shape}"
    )
    alpha = 3 * d
    generator = torch.Generator(device=real.device).manual_seed(rng_seed)
    u_proj = torch.randn(
        (n_projections, d), generator=generator, dtype=real.dtype, device=real.device)
    u_proj = u_proj / u_proj.norm(dim=1, keepdim=True)

    if projection_batch_size is None:
        projection_batch_size = n_projections
    dists = []
    for u_batch in torch.split(u_proj, projection_batch_size):
        # (N, batch): sort each projection over the samples
        x_proj = torch.sort(real @ u_batch.T, dim=0).values
        y_proj = torch.sort(fake @ u_batch.T, dim=0).values
        dists.append(torch.mean((x_proj - y_proj) ** 2, dim=0))

    return alpha * torch.mean(torch.cat(dists))


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


@register_distance
def mse(real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    """Sum, over the batch, of the per-sample mean squared error
    between two batches of feature vectors.

    Unlike the population-level distances above (frechet_distance,
    sliced_wasserstein_distance, ...), which compare two whole
    distributions in one call, this compares paired samples and
    returns a batch-summed value meant to be accumulated across
    multiple calls and divided by the total count (see
    :class:`texture_metrics.samples.SampleDistance`).

    Args:
        real (torch.Tensor): real features, shape (B, ...).
        fake (torch.Tensor): synthetic features, shape (B, ...).

    Returns:
        torch.Tensor: scalar, summed over the batch.
    """
    return torch.mean(
        (fake - real).flatten(start_dim=1) ** 2, dim=-1
    ).sum(dim=0)


@register_distance
def l1(real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    """Sum, over the batch, of the per-sample mean absolute error
    between two batches of feature vectors. See :func:`mse`.

    Args:
        real (torch.Tensor): real features, shape (B, ...).
        fake (torch.Tensor): synthetic features, shape (B, ...).

    Returns:
        torch.Tensor: scalar, summed over the batch.
    """
    return torch.mean(
        (fake - real).flatten(start_dim=1).abs(), dim=-1
    ).sum(dim=0)

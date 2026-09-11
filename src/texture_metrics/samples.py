"""Individual (per-sample, paired target/synth) metrics and registry."""

from functools import wraps
from pathlib import Path
import time
from typing import Callable, Iterable, Optional, Tuple

import torch
from torchmetrics import Metric
from torch.utils.data import DataLoader

from .criteria import weighted_feature_distance
from .criteria import gradients, fourier, optimal_transport
from .criteria.cnn import CNN, RandomTripletDataset
from .transforms import get_transform


class StyleDistance(Metric):
    def __init__(
        self,
        cnn: Optional[CNN | dict] = None,
        features: Iterable[str] = ["mean", "gram", "covariance"],
        contributions: bool = True,
        transform: Optional[Callable] = None,
        transform_name: Optional[str] = None,
        transform_path: Optional[str | Path] = None,
        name: str = "style_distance",
        compile: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.name = name
        self.features = features
        self.contributions = contributions
        if transform is not None:
            self.transform = transform
        else:
            self.transform = get_transform(transform_name, transform_path)
        self.kwargs = kwargs

        if isinstance(cnn, dict):
            cnn = CNN(**cnn)
        self.cnn = cnn

        if compile:
            self.cnn.compile()

        dummy_tensor = torch.randn(1, 3, 256, 256)
        dummy_output = cnn(dummy_tensor)
        self.num_levels = len(dummy_output)
        for f in self.features:
            self.add_state(f, default=torch.tensor(0.0), dist_reduce_fx="sum")
            if self.contributions:
                for i in range(self.num_levels):
                    self.add_state(
                        f"{f}_{i}", default=torch.tensor(0.0), dist_reduce_fx="sum"
                    )
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("time", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")

    def update(
        self,
        target: torch.Tensor,
        synth: torch.Tensor,
        cnn: Optional[CNN] = None,
    ):
        start_time = time.time()
        cnn = cnn or self.cnn

        target_outputs = cnn(self.transform(target))
        synth_outputs = cnn(self.transform(synth))

        for f in self.features:
            res = weighted_feature_distance(
                synth_outputs,
                target_outputs,
                f,
                weights=cnn.layers_weights,
                contributions=self.contributions,
                **self.kwargs,
            ).sum(dim=0)
            if self.contributions:
                setattr(self, f, getattr(self, f) + res[-1])
                for i in range(self.num_levels):
                    setattr(self, f"{f}_{i}", getattr(self, f"{f}_{i}") + res[i])
            else:
                setattr(self, f, getattr(self, f) + res)

        total_time = time.time() - start_time
        self.time += total_time
        self.count += target.size(0)

    def compute(self) -> dict[str, torch.Tensor]:
        metrics = {}
        for f in self.features:
            metrics[f] = getattr(self, f) / self.count
            if self.contributions:
                for i in range(self.num_levels):
                    metrics[f"{f}_{i}"] = getattr(self, f"{f}_{i}") / self.count

        return metrics


class StochasticStyleDistance(StyleDistance):
    def __init__(
        self,
        cnn=None,
        features=["mean", "gram", "covariance"],
        contributions: bool = True,
        transform: Optional[Callable] = None,
        transform_name: Optional[str] = None,
        transform_path: Optional[str | Path] = None,
        name="stochastic_style_distance",
        compile=True,
        batch_size: int = 1,
        **kwargs,
    ):
        super().__init__(
            cnn, features, contributions, transform, transform_name, transform_path, name, compile, **kwargs
        )
        self.batch_size = batch_size

    def update(
        self,
        target: torch.Tensor,
        synth: torch.Tensor,
        cnn: Optional[CNN] = None,
    ):
        start_time = time.time()
        cnn = cnn or self.cnn
        b, c, h, w = target.shape
        triplet_generator = DataLoader(
            RandomTripletDataset(c), batch_size=self.batch_size
        )

        results = torch.zeros(
            len(self.features),
            1 + self.contributions * self.num_levels,
            device=target.device,
        )
        for channels in triplet_generator:
            target_outputs = cnn(target[..., channels, :, :].reshape(-1, 3, h, w))
            synth_outputs = cnn(synth[..., channels, :, :].reshape(-1, 3, h, w))
            for i, f in enumerate(self.features):
                results[i].add_(
                    weighted_feature_distance(
                        synth_outputs,
                        target_outputs,
                        f,
                        weights=cnn.layers_weights,
                        contributions=self.contributions,
                        **self.kwargs,
                    ).sum(dim=0)
                )

        results = results / len(triplet_generator)
        for i, f in enumerate(self.features):
            if self.contributions:
                setattr(self, f, getattr(self, f) + results[i, -1])
                for j in range(self.num_levels):
                    setattr(self, f"{f}_{j}", getattr(self, f"{f}_{j}") + results[i, j])
            else:
                setattr(self, f, getattr(self, f) + results[i])

        total_time = torch.tensor(time.time() - start_time)
        self.time += total_time
        self.count += target.size(0)


_metric_dict = dict()


def is_valid_metric(metric):
    return metric in _metric_dict


def list_valid_metrics():
    return list(_metric_dict.keys())


def register_metric(func: Callable):
    assert callable(func)

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        with torch.no_grad():
            value = func(*args, **kwargs)
        total_time = time.time() - start_time
        return value, total_time

    _metric_dict[func.__name__] = wrapper

    return wrapper


# ------------------------------- Distrbutions ----------------------------- #


def distribution_distances(
    target: torch.Tensor,
    synth: torch.Tensor,
    nslice: Optional[int] = 1,
    batch_size: Optional[int] = None,
):
    """Computes distribution distances (band-wise Wasserstein
    distance and SWD) between target and synthetic images.

    Args:
        target (torch.Tensor): target image.
        synth (torch.Tensor): synthetic image.
        nslice (int): Number of slices for SWD

    Returns:
        dict: dictionnary containing the distribution distances.
    """
    hist_dist = optimal_transport.histogram_loss1D(target, synth).sqrt().sum(dim=0)
    return {
        "swd": optimal_transport.sliced_wasserstein_distance(
            target, synth, nslice=nslice, batch_size=batch_size
        ).sum(dim=0),
        **dict([(f"band_{i}", hist_dist[i]) for i in range(target.size(-3))]),
    }


@register_metric
def sliced_wasserstein_distance_image(
    target: torch.Tensor,
    synth: torch.Tensor,
    nslice: Optional[int] = 1,
    batch_size: Optional[int] = None,
):
    """Computes Sliced Wasserstein Distance (SWD) between target and
    synthetic images.

    Args:
        target (torch.Tensor): target image.
        synth (torch.Tensor): synthetic image.
        options (MetricsOptions): metrics options.

    Returns:
        Number: SWD
    """
    return optimal_transport.sliced_wasserstein_distance(
        target, synth, nslice=nslice, batch_size=batch_size
    ).sum(dim=0)


@register_metric
def histograms(
    target: torch.Tensor,
    synth: torch.Tensor,
    nslice: Optional[int] = 1,
    batch_size: Optional[int] = None,
):
    """Computes histogram distances.

    Args:
        target (torch.Tensor): target image.
        synth (torch.Tensor): synthetic image.
        options (MetricsOptions): metrics options.

    Returns:
        dict: dictionnary containing histogram distances.
    """
    return distribution_distances(target, synth, nslice=nslice, batch_size=batch_size)


def get_stats(tnsr: torch.Tensor, cholesky=False) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute first and second order statistics of an image.

    Args:
        tnsr (torch.Tensor): input image
        cholesky (bool, optional): Computes and return Cholesky
            decomposition of the covariance matrix.
            Defaults to False.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: mean and
            covariance/Cholesky decomposition
    """
    tnsr = tnsr.flatten(start_dim=-2)
    mu = tnsr.mean(dim=-1, keepdim=True)
    tnsr = tnsr - mu
    cov = tnsr @ tnsr.transpose(-1, -2) / tnsr.shape[-1]
    if cholesky:
        l = torch.linalg.cholesky(cov)
        return mu.squeeze(-1), l
    else:
        return mu.squeeze(-1), cov


@register_metric
def color_statistics(target: torch.Tensor, synth: torch.Tensor):
    """Computes color statistics distances (mean, cov, RX).

    Args:
        target (torch.Tensor): target image.
        synth (torch.Tensor): synthetic image.
        options (MetricsOptions): metrics options.

    Returns:
        dict: dictionnary containing color statistics distances.
    """
    mut, covt = get_stats(target)
    mus, covs = get_stats(synth)
    bure_distance = torch.sqrt(
        torch.mean((mut - mus) ** 2, dim=-1)
        + optimal_transport.bure_distance(covt, covs)
    )
    return {
        "mean": torch.mean((mut - mus) ** 2, dim=-1).sum(dim=0),
        "covariance": torch.mean((covt - covs) ** 2, dim=(-1, -2)).sum(dim=0),
        "RX": bure_distance.sum(dim=0),
    }


# ----------------------------- Fourier spectra ---------------------------- #


@register_metric
def spectral_radial_distance(target: torch.Tensor, synth: torch.Tensor):
    """Computes L-2 distance on azimuthal spectra (mean and band-wise).

    Args:
        target (torch.Tensor): target image.
        synth (torch.Tensor): synthetic image.
        options (MetricsOptions): metrics options.

    Returns:
        dict: dictionnary containing mean and band-wise radial
            spectral distances.
    """
    names = [f"band_{i}" for i in range(target.size(-3))]
    dist_mean = fourier.spectral_radial_distance(
        target.mean(dim=-3), synth.mean(dim=-3)
    ).sqrt().sum(dim=0)
    dist_band = fourier.spectral_radial_distance(target, synth).sqrt().sum(dim=0)
    return {"mean": dist_mean, **dict(zip(names, dist_band))}


# -------------------------------- Gradients ------------------------------- #


@register_metric
def gradients_magnitude_distance(
    target: torch.Tensor,
    synth: torch.Tensor,
    nslice: Optional[int] = 1,
    batch_size: Optional[int] = None,
):
    """Computes gradients distribution distances (along x and y axis
    and magnitude).

    Args:
        target (torch.Tensor): target image.
        synth (torch.Tensor): synthetic image.
        options (MetricsOptions): metrics options.

    Returns:
        dict: dictionnary containing gradients distances.
    """
    dt = gradients.image_gradient(target, result='mag')
    ds = gradients.image_gradient(synth, result='mag')
    return distribution_distances(dt, ds, nslice=nslice, batch_size=batch_size)


class SimpleDistance(Metric):
    def __init__(self, dist_fn: Callable | str, name: Optional[str], kwargs: dict = {}):
        super().__init__()

        if isinstance(dist_fn, str):
            self.dist_fn = _metric_dict[dist_fn]

        if name is None:
            name = dist_fn.__name__
        self.name = name

        self.kwargs = kwargs

        self.add_state("distance", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("time", default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, target: torch.Tensor, synth: torch.Tensor):
        value, time = self.dist_fn(target, synth, **self.kwargs)
        self.distance += value
        self.count += target.size(0)
        self.time += time

    def compute(self) -> torch.Tensor:
        """Compute the final metric value.

        Returns:
            torch.Tensor: the average distance over all samples.
        """
        return self.distance / self.count

class DictDistance(Metric):
    def __init__(self, dist_fn: Callable | str, name: Optional[str], nbands: int = 3, kwargs: dict = {}):
        super().__init__()

        if isinstance(dist_fn, str):
            self.dist_fn = _metric_dict[dist_fn]

        if name is None:
            name = self.dist_fn.__name__
        self.name = name

        self.kwargs = kwargs

        dummy_target = torch.randn(1, nbands, 64, 64)
        dummy_synth = torch.randn(1, nbands, 64, 64)
        dummy_output, _ = self.dist_fn(dummy_target, dummy_synth)
        self.keys = dummy_output.keys()
        for k in self.keys:
            self.add_state(k, default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("time", default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, target: torch.Tensor, synth: torch.Tensor):
        values, time = self.dist_fn(target, synth, **self.kwargs)
        for k,v in values.items():
            setattr(self, k, getattr(self, k) + v)
        self.count += target.size(0)
        self.time += time

    def compute(self) -> torch.Tensor:
        """Compute the final metric value.

        Returns:
            torch.Tensor: the average distance over all samples.
        """
        results = {}
        for k in self.keys:
            results[k] = getattr(self, k) / self.count
        return results

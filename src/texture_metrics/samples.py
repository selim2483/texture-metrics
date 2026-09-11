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
from .distances import _dist_dict
from .representations import _repr_dict, flatten
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


# ------------------------ Representation + distance ----------------------- #


class SampleDistance(Metric):
    """Per-sample metric, built either from a registered full metric
    function, or by composing a representation function with a
    distance function.

    Two mutually exclusive ways to build one:

    - ``metric_fn``: a function (or its name, looked up in this
      module's ``_metric_dict``) taking ``(target, synth, **kwargs)``
      directly and returning a value (scalar tensor or dict of
      tensors) -- e.g. ``"color_statistics"``, ``"histograms"``.
    - ``dist_fn`` (+ optional ``repr_fn``): ``repr_fn`` first maps
      ``target``/``synth`` to representations (defaults to spatial
      flattening, turning images into feature vectors), then
      ``dist_fn`` compares the two representations. ``dist_fn`` is
      looked up (if given as a string) in
      ``texture_metrics.distances._dist_dict`` -- so both simple
      vector distances (``"mse"``, ``"l1"``) and population-level ones
      (``"frechet_distance"``, ``"sliced_wasserstein_distance"``, ...)
      can be used here, applied per-batch rather than over the whole
      accumulated population. ``repr_fn`` is looked up in
      ``texture_metrics.representations._repr_dict`` if given as a
      string.

    Whether the underlying function returns a scalar or a dict of
    scalars is auto-detected from a dummy call at construction time,
    so both cases (mirroring the old ``SimpleDistance``/
    ``DictDistance`` split) are handled by the same class.

    Args:
        metric_fn (Optional[Callable | str]): full metric function, or
            its registered name.
        dist_fn (Optional[Callable | str]): distance function, or its
            registered name. Required if ``metric_fn`` is not given.
        repr_fn (Optional[Callable | str]): representation function,
            or its registered name. Only used together with
            ``dist_fn``; defaults to :func:`texture_metrics.representations.flatten`.
        metric_fn_kwargs (dict): keyword arguments for ``metric_fn``.
        repr_fn_kwargs (dict): keyword arguments for ``repr_fn``.
        dist_fn_kwargs (dict): keyword arguments for ``dist_fn``.
        name (Optional[str]): metric name. Defaults to
            ``metric_fn.__name__`` or
            ``f"{repr_fn.__name__}/{dist_fn.__name__}"``.
        nchannels (int): number of channels for the dummy target/synth
            images used to probe, at construction time, whether the
            metric produces a scalar or a dict of scalars. Some
            metrics' key sets depend on the channel count (e.g.
            ``histograms``' per-band keys), so this must match the
            number of channels the metric will actually be run on.
    """

    def __init__(
        self,
        metric_fn: Optional[Callable | str] = None,
        dist_fn: Optional[Callable | str] = None,
        repr_fn: Optional[Callable | str] = None,
        metric_fn_kwargs: dict = {},
        repr_fn_kwargs: dict = {},
        dist_fn_kwargs: dict = {},
        name: Optional[str] = None,
        nchannels: int = 3,
    ):
        super().__init__()

        self.metric_fn = None
        self.repr_fn = None
        self.dist_fn = None

        if metric_fn is not None:
            if isinstance(metric_fn, str):
                metric_fn = _metric_dict[metric_fn]
            self.metric_fn = metric_fn
            self.metric_fn_kwargs = metric_fn_kwargs or {}
            if name is None:
                name = metric_fn.__name__
        else:
            assert dist_fn is not None, (
                "SampleDistance requires either `metric_fn` or `dist_fn`."
            )
            if isinstance(dist_fn, str):
                dist_fn = _dist_dict[dist_fn]
            self.dist_fn = dist_fn
            self.dist_fn_kwargs = dist_fn_kwargs or {}

            if repr_fn is None:
                repr_fn = flatten
            elif isinstance(repr_fn, str):
                repr_fn = _repr_dict[repr_fn]
            self.repr_fn = repr_fn
            self.repr_fn_kwargs = repr_fn_kwargs or {}

            if name is None:
                name = f"{repr_fn.__name__}/{dist_fn.__name__}"
        self.name = name

        # Probe with dummy tensors: the underlying function may return
        # a plain scalar (e.g. sliced_wasserstein_distance_image,
        # frechet_distance) or a dict of scalars (e.g.
        # color_statistics, per_bin_wasserstein_distance) -- states
        # must be registered up front either way. `nchannels` matters
        # here: some metrics' key sets depend on it (e.g. histograms'
        # per-band keys track target.size(-3)), so the probe must use
        # the same channel count the metric will really be run on.
        #
        # Sizing otherwise differs by path: metric_fn operates within
        # a single image (spatial statistics), so a lone 64x64 dummy
        # is cheap and matches real usage. The dist_fn path may reduce
        # the representation to a population (e.g. frechet_distance's
        # covariance + eigendecomposition, cost O(D^3) in the
        # representation dimension D) -- with the default `flatten`
        # repr_fn, a 64x64 probe image would blow that dimension up to
        # nchannels*64*64 for nothing, and a single sample makes the
        # covariance degenerate besides. Use a small batch of small
        # images instead, cheap regardless of repr_fn/dist_fn.
        if self.metric_fn is not None:
            dummy_target = torch.randn(1, nchannels, 64, 64)
            dummy_synth = torch.randn(1, nchannels, 64, 64)
        else:
            dummy_target = torch.randn(4, nchannels, 4, 4)
            dummy_synth = torch.randn(4, nchannels, 4, 4)
        dummy_value, _ = self._compute(dummy_target, dummy_synth)
        self._is_dict = isinstance(dummy_value, dict)
        if self._is_dict:
            self.keys = list(dummy_value.keys())
            for k in self.keys:
                self.add_state(k, default=torch.tensor(0.0), dist_reduce_fx="sum")
        else:
            self.add_state("distance", default=torch.tensor(0.0), dist_reduce_fx="sum")

        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("time", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def _compute(
        self, target: torch.Tensor, synth: torch.Tensor
    ) -> Tuple[torch.Tensor | dict[str, torch.Tensor], float]:
        """Runs the metric_fn or repr_fn+dist_fn path.

        Returns:
            Tuple[torch.Tensor | dict[str, torch.Tensor], float]:
                the value (scalar or dict of scalars) and the elapsed
                time.
        """
        if self.metric_fn is not None:
            return self.metric_fn(target, synth, **self.metric_fn_kwargs)

        start_time = time.time()
        target_repr = self.repr_fn(target, **self.repr_fn_kwargs)
        synth_repr = self.repr_fn(synth, **self.repr_fn_kwargs)
        value = self.dist_fn(target_repr, synth_repr, **self.dist_fn_kwargs)
        return value, time.time() - start_time

    def update(self, target: torch.Tensor, synth: torch.Tensor):
        """Update the metric state with a batch of paired images.

        Args:
            target (torch.Tensor): target images, shape (B, C, H, W).
            synth (torch.Tensor): synthetic images, shape (B, C, H, W).
        """
        value, elapsed = self._compute(target, synth)
        if self._is_dict:
            for k, v in value.items():
                setattr(self, k, getattr(self, k) + v)
        else:
            self.distance = self.distance + value
        self.count += target.size(0)
        self.time = self.time + elapsed

    def compute(self) -> torch.Tensor | dict[str, torch.Tensor]:
        """Compute the final metric value.

        Returns:
            torch.Tensor | dict[str, torch.Tensor]: the average
                distance (or dict of average distances) over all
                samples.
        """
        if self._is_dict:
            return {k: getattr(self, k) / self.count for k in self.keys}
        return self.distance / self.count

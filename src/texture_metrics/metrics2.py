import csv
from datetime import datetime
from functools import wraps
import json
from pathlib import Path
import time
from typing import Callable, Iterable, List, Optional

from texture_metrics.utils.logging import progress_bar
from texture_metrics.utils.seed import (
    collect_rng_states,
    seed_everything,
    set_rng_states,
)

import torch
from torchmetrics import Metric
from torch.utils.data import DataLoader

from .criteria import weighted_feature_distance
from .criteria import gradients, fourier, optimal_transport
from .criteria.cnn import CNN, RandomTripletDataset
from .transforms import get_stats

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as file:
        json.dump(payload, file, indent=2)


class StyleDistance(Metric):
    def __init__(
        self,
        cnn: Optional[CNN | dict] = None,
        features: Iterable[str] = ["mean", "gram", "covariance"],
        contributions: bool = True,
        transform: Optional[str | Callable] = None,
        name: str = "style_distance",
        compile: bool = True,
        **kwargs,
    ):
        self.name = name
        self.features = features
        self.contributions = contributions
        self.transform = transform
        self.kwargs = kwargs

        if isinstance(cnn, dict):
            cnn = CNN(**cnn)
        self.cnn = cnn

        if compile:
            self.cnn.compile()

        dummy_tensor = torch.randn(1, 3, 128, 128)
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
        self.add_state("time", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self,
        target: torch.Tensor,
        synth: torch.Tensor,
        cnn: Optional[CNN] = None,
    ):
        start_time = time.time()
        cnn = cnn or self.cnn

        target_outputs = cnn(target)
        synth_outputs = cnn(synth)

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
                getattr(self, f) += res[-1]
                for i in range(self.num_levels):
                    getattr(self, f"{f}_{i}") += res[i]
            else:
                getattr(self, f) += res

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
        contributions=True,
        transform=None,
        name="style_distance",
        compile=True,
        batch_size: int = 1,
        **kwargs,
    ):
        super().__init__(
            cnn, features, contributions, transform, name, compile, **kwargs
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
        triplet_generator = DataLoader(
            RandomTripletDataset(target.shape[-3]), batch_size=self.batch_size
        )

        results = torch.zeros(
            len(self.features),
            1 + self.contributions * self.num_levels,
            device=target.device,
        )
        for channels in triplet_generator:
            target_outputs = cnn(target[..., channels, :, :].squeeze(0))
            synth_outputs = cnn(synth[..., channels, :, :].squeeze(0))
            for i, f in enumerate(self.features):
                results[i].add_(
                    channels.size(0)
                    * weighted_feature_distance(
                        synth_outputs,
                        target_outputs,
                        f,
                        weights=cnn.layers_weights,
                        contributions=self.contributions,
                        **self.kwargs,
                    )
                )

        results = results.sum(dim=0) / len(triplet_generator)
        for f in self.features:
            if self.contributions:
                getattr(self, f) += results[-1]
                for i in range(self.num_levels):
                    getattr(self, f"{f}_{i}") += results[i]
            else:
                getattr(self, f) += results

        total_time = time.time() - start_time
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
    nslice: Optional[int],
    batch_size: Optional[int],
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
    return {
        "swd": optimal_transport.sliced_wasserstein_distance(
            target, synth, nslice=nslice, batch_size=batch_size
        ).tolist(),
        **dict(
            zip(
                [f"band_{i}" for i in range(target.size(-3))],
                optimal_transport.histogram_loss1D(target, synth).sqrt().T.tolist(),
            )
        ),
    }


@register_metric
def sliced_wasserstein_distance(
    target: torch.Tensor,
    synth: torch.Tensor,
    nslice: Optional[int],
    batch_size: Optional[int],
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
    ).tolist()


@register_metric
def histograms(
    target: torch.Tensor,
    synth: torch.Tensor,
    nslice: Optional[int],
    batch_size: Optional[int],
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
        "mean": torch.mean((mut - mus) ** 2, dim=-1).tolist(),
        "covariance": torch.mean((covt - covs) ** 2, dim=(-1, -2)).tolist(),
        "RX": bure_distance.tolist(),
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
    ).sqrt()
    dist_band = fourier.spectral_radial_distance(target, synth).sqrt()
    return {"mean": dist_mean.tolist(), **dict(zip(names, dist_band.T.tolist()))}


# -------------------------------- Gradients ------------------------------- #


@register_metric
def gradients_distance(
    target: torch.Tensor,
    synth: torch.Tensor,
    nslice: Optional[int],
    batch_size: Optional[int],
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
    dt_x, dt_y, dt = gradients.image_gradient(target)
    ds_x, ds_y, ds = gradients.image_gradient(synth)
    return {
        "dx": distribution_distances(dt_x, ds_x, nslice=nslice, batch_size=batch_size),
        "dy": distribution_distances(dt_y, ds_y, nslice=nslice, batch_size=batch_size),
        "dmag": distribution_distances(dt, ds, nslice=nslice, batch_size=batch_size),
    }


class SimpleDistance(Metric):
    def __init__(self, dist_fn: Callable | str, name: Optional[str], kwargs: dict = {}):
        if isinstance(dist_fn, str):
            self.dist_fn = _metric_dict[dist_fn]

        if name is None:
            name = dist_fn.__name__
        self.name = name

        self.kwargs = kwargs

        self.add_state("distance", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("time", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, target: torch.Tensor, synth: torch.Tensor):
        value, time = self.dist_fn(target, synth, **self.kwargs)
        self.distance += target.size(0) * value
        self.count += target.size(0)
        self.time += time

    def compute(self) -> torch.Tensor:
        """Compute the final metric value.

        Returns:
            torch.Tensor: the average distance over all samples.
        """
        return self.distance / self.count


def compute_metrics(metrics: List[Metric], reset: bool = True) -> dict:
    results = {}
    for metric in metrics:
        metric_value = metric.compute()
        if isinstance(metric_value, torch.Tensor):
            metric_value = metric_value.item()
        elif isinstance(metric_value, dict):
            metric_value = {k: v.item() for k, v in metric_value.items()}
        results[metric.name] = metric_value
        if reset:
            metric.reset()
    return results


def metrics_loop(
    model: torch.nn.Module,
    metrics: List[Metric],
    loader: torch.utils.data.DataLoader,
    nimg: Optional[int] = None,
    seed: Optional[int] = None,
    enable_progress_bar: bool = True,
    **kwargs,
):
    _rng_states = collect_rng_states()
    seed_everything(seed)
    model.eval()
    start_time = datetime.datetime.now()
    img_count = 0

    with progress_bar(
        enable_progress_bar=enable_progress_bar, global_rank=0
    ) as progress:
        if progress is not None:
            task_id = progress.add_task(
                "Metrics/Generation",
                total=len(loader) if hasattr(loader, "__len__") else None,
            )

            while img_count < nimg if nimg is not None else True:
                for batch in loader:
                    if nimg is not None and img_count >= nimg:
                        break

                    output = model(batch)

                    for metric in metrics:
                        metric.update(output["sample"], output["target"])

                    if progress is not None and task_id is not None:
                        progress.update(task_id, advance=1)

                    img_count += output["target"].size(0)

                if nimg is None:
                    break  # Exit the loop if nimg is not specified

        print(f"[info] Total images processed for metrics: {img_count}")
        print(f"[info] Computing final metrics...")
        results = compute_metrics(metrics, reset=True)

    set_rng_states(_rng_states)
    torch.cuda.empty_cache()

    endtime = datetime.datetime.now()
    results["starttime"] = str(start_time.strftime("%Y-%m-%d %H:%M:%S"))
    results["endtime"] = str(endtime.strftime("%Y-%m-%d %H:%M:%S"))
    results["duration"] = str(endtime - start_time)

    return results


def save_metrics(results: dict, save_dir: str, output_name: str):
    """
    Save the metrics results to a YAML file in the specified directory.
    """
    # ── Save JSON ────────────────────────────────────────────────────────────
    json_path = save_dir / f"{output_name}_metrics.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[info] JSON saved → {json_path}")

    # ── Save CSV ─────────────────────────────────────────────────────────────
    csv_path = save_dir / f"{output_name}_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        for metric_key, value in sorted(results.items()):
            writer.writerow({"metric": metric_key, "value": value})

    print(f"[info] CSV  saved → {csv_path}")

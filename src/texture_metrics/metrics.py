"""Metric-agnostic evaluation loop and result utilities.

Shared by both individual (per-sample) and distribution metrics: anything
implementing the ``torchmetrics.Metric`` interface (``update``/``compute``)
can be passed to :func:`metrics_loop`, regardless of which registry
(individual.py's ``_metric_dict`` or distributions.py's population-level
metrics) it comes from.
"""

import csv
from datetime import datetime
import json
from pathlib import Path
from typing import List, Optional

from texture_metrics.utils.logging import progress_bar
from texture_metrics.utils.seed import (
    collect_rng_states,
    seed_everything,
    set_rng_states,
)

import torch
from torchmetrics import Metric

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as file:
        json.dump(payload, file, indent=2)


def compute_metrics(metrics: List[Metric], reset: bool = True) -> dict:
    results = {}
    for metric in metrics:
        metric_value = metric.compute()
        if isinstance(metric_value, torch.Tensor):
            metric_value = metric_value.item()
        elif isinstance(metric_value, dict):
            metric_value = {k: v.item() for k, v in metric_value.items()}
        results[metric.name] = {"value": metric_value, "time": metric.time.item()}
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
    """Runs a list of metrics over a loader given a model.

    Args:
        model (torch.nn.Module): callable taking a batch of real images
            and returning a dict with a ``"sample"`` key (synthetic
            images) and a ``"target"`` key (the corresponding real
            images).
        metrics (List[Metric]): metrics to update, each following the
            ``torchmetrics.Metric`` interface (``update(sample, target)``,
            ``compute()``). Can mix individual (per-sample) and
            distribution (population-level) metrics.
        loader (torch.utils.data.DataLoader): data loader yielding
            batches for ``model``.
        nimg (Optional[int]): number of images to process. If it
            exceeds the loader's size, the loader is cycled. If None,
            a single pass over the loader is performed.
        seed (Optional[int]): random seed for reproducibility.
        enable_progress_bar (bool): whether to display a progress bar.

    Returns:
        dict: metric results, keyed by metric name, plus timing/count
            metadata.
    """
    _rng_states = collect_rng_states()
    seed_everything(seed)
    model.eval()
    start_time = datetime.now()
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

                    output = model(batch, **kwargs)

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

    endtime = datetime.now()
    results["starttime"] = str(start_time.strftime("%Y-%m-%d %H:%M:%S"))
    results["endtime"] = str(endtime.strftime("%Y-%m-%d %H:%M:%S"))
    results["duration"] = str(endtime - start_time)
    results["number_images"] = img_count

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

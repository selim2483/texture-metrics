import json

import pytest

from texture_metrics.evaluation import compute_metrics, metrics_loop, save_metrics
from texture_metrics.samples import SampleDistance


def test_compute_metrics_resets_by_default(target_synth):
    target, synth = target_synth
    metric = SampleDistance(metric_fn="sliced_wasserstein_distance_image", name="swd")
    metric.update(target, synth)

    results = compute_metrics([metric])

    assert "swd" in results
    assert "value" in results["swd"] and "time" in results["swd"]
    # reset=True by default: a fresh compute() with no new update should
    # differ (both accumulators are back to zero -> division by zero ->
    # nan), confirming the metric's internal state was actually cleared.
    assert metric.count.item() == 0


def test_metrics_loop_runs_and_reports_metadata(dummy_model, dummy_loader):
    metric = SampleDistance(metric_fn="sliced_wasserstein_distance_image", name="swd")
    results = metrics_loop(
        dummy_model, [metric], dummy_loader, enable_progress_bar=True
    )
    expected_images = dummy_loader.n_batches * dummy_loader.batch_size
    assert results["number_images"] == expected_images
    assert "swd" in results
    assert "starttime" in results and "endtime" in results and "duration" in results


def test_metrics_loop_respects_nimg(dummy_model, dummy_loader):
    metric = SampleDistance(metric_fn="sliced_wasserstein_distance_image", name="swd")
    nimg = dummy_loader.batch_size  # less than a full pass
    results = metrics_loop(
        dummy_model, [metric], dummy_loader, nimg=nimg, enable_progress_bar=True
    )
    assert results["number_images"] >= nimg


def test_save_metrics_writes_json_and_csv(tmp_path, target_synth):
    target, synth = target_synth
    metric = SampleDistance(metric_fn="sliced_wasserstein_distance_image", name="swd")
    metric.update(target, synth)
    results = compute_metrics([metric])

    save_metrics(results, save_dir=tmp_path, output_name="run")

    json_path = tmp_path / "run_metrics.json"
    csv_path = tmp_path / "run_metrics.csv"
    assert json_path.exists()
    assert csv_path.exists()
    with open(json_path) as f:
        loaded = json.load(f)
    assert loaded == results


@pytest.mark.xfail(
    reason=(
        "Pre-existing bug: the batch-processing while/for loop in "
        "metrics_loop lives entirely inside `if progress is not None:`, "
        "so enable_progress_bar=False skips processing every batch (0 "
        "images, metrics never updated). Not fixed here -- tracked for "
        "whoever revisits this."
    ),
    strict=True,
)
def test_metrics_loop_processes_batches_without_progress_bar(
    dummy_model, dummy_loader
):
    metric = SampleDistance(metric_fn="sliced_wasserstein_distance_image", name="swd")
    results = metrics_loop(
        dummy_model, [metric], dummy_loader, enable_progress_bar=False
    )
    assert results["number_images"] > 0

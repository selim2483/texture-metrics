"""End-to-end regression tests exercising sample and distribution
metrics together through the shared metrics_loop -- the scenario the
package's whole split (evaluation.py vs samples.py/distributions.py)
exists to support."""

from texture_metrics import (
    SampleDistance,
    DistributionDistance,
    CNNDistributionDistance,
    FID,
    metrics_loop,
)
from texture_metrics.representations import RadialProfileExtractor


def test_mixed_sample_and_distribution_metrics_share_the_loop(
    dummy_model, dummy_loader, tiny_cnn
):
    metrics = [
        SampleDistance(metric_fn="sliced_wasserstein_distance_image", name="swd_image"),
        SampleDistance(repr_fn="color_mean", dist_fn="mse"),
        # dist_fn borrowed from distances.py's population-level registry,
        # applied here per-batch via SampleDistance's default flatten repr_fn.
        SampleDistance(dist_fn="frechet_distance", name="flatten/frechet_per_batch"),
        DistributionDistance(repr_fn="color_mean", dist_fn="frechet_distance"),
        CNNDistributionDistance(
            cnn=tiny_cnn,
            dist_fn="sliced_wasserstein_distance",
            dist_fn_kwargs={"nslice": 8, "bslice": 4},
            input_image_size=(3, dummy_loader.size, dummy_loader.size),
            compile=False,
        ),
        FID(
            feature=RadialProfileExtractor(bin_size=2),
            input_img_size=(3, dummy_loader.size, dummy_loader.size),
            compile=False,
            value_range=(-3.0, 3.0),
        ),
    ]

    results = metrics_loop(dummy_model, metrics, dummy_loader, enable_progress_bar=True)

    for metric in metrics:
        entry = results[metric.name]
        assert "value" in entry
        assert "time" in entry
        assert entry["time"] >= 0.0

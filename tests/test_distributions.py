import torch

from texture_metrics.distributions import (
    DistributionDistance,
    CNNDistributionDistance,
    FID,
)
from texture_metrics.representations import RadialProfileExtractor


def test_distribution_distance_update_compute(target_synth):
    target, synth = target_synth
    metric = DistributionDistance(repr_fn="color_mean", dist_fn="frechet_distance")
    metric.update(synth, target)
    value = metric.compute()
    assert torch.is_tensor(value)
    assert metric.time.item() >= 0.0


def test_distribution_distance_near_zero_for_identical_populations(target_synth):
    target, _ = target_synth
    metric = DistributionDistance(repr_fn="color_mean", dist_fn="frechet_distance")
    metric.update(target, target)
    assert metric.compute().item() < 1e-4


def test_cnn_distribution_distance_update_compute(tiny_cnn, target_synth):
    target, synth = target_synth
    metric = CNNDistributionDistance(
        cnn=tiny_cnn,
        dist_fn="sliced_wasserstein_distance",
        dist_fn_kwargs={"nslice": 8, "bslice": 4},
        input_image_size=tuple(target.shape[1:]),
        compile=False,
    )
    metric.update(synth, target)
    result = metric.compute()
    assert set(result.keys()) == {"mean_0", "mean_1", "std_0", "std_1"}
    assert metric.time.item() >= 0.0


def test_fid_update_compute_with_lightweight_feature_extractor(target_synth):
    target, synth = target_synth
    metric = FID(
        feature=RadialProfileExtractor(bin_size=2),
        input_img_size=tuple(target.shape[1:]),
        compile=False,
        value_range=(-3.0, 3.0),
    )
    metric.update(synth, target)
    value = metric.compute()
    assert torch.is_tensor(value)
    assert metric.time.item() >= 0.0


def test_max_real_samples_caps_both_real_and_fake(target_synth):
    target, synth = target_synth
    n = target.shape[0]
    metric = DistributionDistance(
        repr_fn="color_mean",
        dist_fn="frechet_distance",
        max_real_samples=n,
    )
    # Two updates: only the first batch's worth of samples should be
    # retained on each side.
    metric.update(synth, target)
    metric.update(synth, target)
    assert sum(t.shape[0] for t in metric.real) == n
    assert sum(t.shape[0] for t in metric.fake) == n

import torch

from texture_metrics.samples import (
    SampleDistance,
    StyleDistance,
    is_valid_metric,
    list_valid_metrics,
)


def test_registry_has_expected_entries():
    for name in [
        "sliced_wasserstein_distance_image",
        "histograms",
        "color_statistics",
        "spectral_radial_distance",
        "gradients_magnitude_distance",
    ]:
        assert is_valid_metric(name)
    assert set(list_valid_metrics()) == {
        "sliced_wasserstein_distance_image",
        "histograms",
        "color_statistics",
        "spectral_radial_distance",
        "gradients_magnitude_distance",
    }


# ------------------------------ metric_fn path ----------------------------- #


def test_sample_distance_metric_fn_scalar(target_synth):
    target, synth = target_synth
    metric = SampleDistance(
        metric_fn="sliced_wasserstein_distance_image", name="swd"
    )
    metric.update(target, synth)
    value = metric.compute()
    assert torch.is_tensor(value)
    assert value.item() >= 0.0
    assert metric.time.item() >= 0.0


def test_sample_distance_metric_fn_scalar_near_zero_for_identical_inputs(
    target_synth,
):
    target, _ = target_synth
    metric = SampleDistance(metric_fn="sliced_wasserstein_distance_image", name="swd")
    metric.update(target, target)
    assert metric.compute().item() < 1e-4


def test_sample_distance_metric_fn_dict(target_synth):
    target, synth = target_synth
    metric = SampleDistance(metric_fn="color_statistics", name="color_stats")
    metric.update(target, synth)
    result = metric.compute()
    assert set(result.keys()) == {"mean", "covariance", "RX"}


def test_sample_distance_metric_fn_default_name():
    metric = SampleDistance(metric_fn="color_statistics")
    assert metric.name == "color_statistics"


def test_sample_distance_nchannels_matches_dummy_probe_keys():
    """histograms' key set (swd + one band_i per channel) depends on
    the channel count -- nchannels must match what update() will
    really see, or the states registered from the dummy probe won't
    line up with what update() tries to accumulate into."""
    nchannels = 5
    metric = SampleDistance(metric_fn="histograms", nchannels=nchannels)
    expected_keys = {"swd"} | {f"band_{i}" for i in range(nchannels)}
    assert set(metric.keys) == expected_keys

    target = torch.randn(2, nchannels, 16, 16)
    synth = target + 0.05 * torch.randn(2, nchannels, 16, 16)
    metric.update(target, synth)
    result = metric.compute()
    assert set(result.keys()) == expected_keys


def test_sample_distance_nchannels_default_is_three():
    metric = SampleDistance(metric_fn="histograms")
    assert set(metric.keys) == {"swd", "band_0", "band_1", "band_2"}


def test_sample_distance_metric_fn_string_with_no_name_is_no_longer_broken():
    # Regression test: SimpleDistance/DictDistance used to crash on
    # (str dist_fn, name=None) -- fixed by this class's unification.
    metric = SampleDistance(metric_fn="sliced_wasserstein_distance_image")
    assert metric.name == "sliced_wasserstein_distance_image"


# --------------------------- repr_fn + dist_fn path ------------------------- #


def test_sample_distance_dist_fn_update_compute(target_synth):
    target, synth = target_synth
    metric = SampleDistance(repr_fn="color_mean", dist_fn="mse")
    metric.update(target, synth)
    value = metric.compute()
    assert torch.is_tensor(value)
    assert value.item() >= 0.0
    assert metric.name == "color_mean/mse"


def test_sample_distance_dist_fn_near_zero_for_identical_inputs(target_synth):
    target, _ = target_synth
    metric = SampleDistance(repr_fn="color_mean", dist_fn="mse")
    metric.update(target, target)
    assert metric.compute().item() == 0.0


def test_sample_distance_dist_fn_accepts_callables_directly(target_synth):
    target, synth = target_synth

    def my_repr(x):
        return x.mean(dim=(-2, -1))

    def my_dist(real, fake):
        return (real - fake).abs().sum()

    metric = SampleDistance(repr_fn=my_repr, dist_fn=my_dist, name="custom")
    metric.update(target, synth)
    assert metric.name == "custom"
    assert torch.is_tensor(metric.compute())


def test_sample_distance_repr_fn_defaults_to_flatten(target_synth):
    target, synth = target_synth
    metric = SampleDistance(dist_fn="mse")
    assert metric.repr_fn.__name__ == "flatten"
    assert metric.name == "flatten/mse"
    metric.update(target, synth)
    assert torch.is_tensor(metric.compute())


def test_sample_distance_dist_fn_falls_back_to_distances_registry(target_synth):
    """dist_fn strings not found in the local pointwise registry are
    looked up in texture_metrics.distances' registry instead."""
    target, synth = target_synth
    metric = SampleDistance(dist_fn="frechet_distance")
    metric.update(target, synth)
    value = metric.compute()
    assert torch.is_tensor(value)
    assert metric.name == "flatten/frechet_distance"


def test_sample_distance_dist_fn_dict_output_from_distances_registry(target_synth):
    """per_bin_wasserstein_distance (distances.py) returns a dict --
    exercises the same auto-detection as metric_fn's dict outputs."""
    target, synth = target_synth
    metric = SampleDistance(repr_fn="color_mean", dist_fn="per_bin_wasserstein_distance")
    metric.update(target, synth)
    result = metric.compute()
    assert isinstance(result, dict)
    assert len(result) > 0


def test_sample_distance_requires_metric_fn_or_dist_fn():
    try:
        SampleDistance()
    except AssertionError:
        pass
    else:
        raise AssertionError("expected SampleDistance() with no args to raise")


# --------------------------------- StyleDistance ---------------------------- #


def test_style_distance_update_compute(tiny_cnn, target_synth):
    target, synth = target_synth
    # transform=Identity() sidesteps the pre-existing get_transform(
    # transform_name, transform_path) construction bug (transform_name/
    # transform_path default to None, and get_transform's real signature
    # is (img, transform), not (transform_name, transform_path)).
    metric = StyleDistance(
        cnn=tiny_cnn,
        features=["mean", "covariance"],
        transform=torch.nn.Identity(),
        compile=False,
    )
    metric.update(target, synth)
    result = metric.compute()
    assert "mean" in result and "covariance" in result

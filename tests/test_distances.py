import torch

from texture_metrics.distances import (
    _dist_dict,
    frechet_distance,
    sliced_wasserstein_distance,
    histogram_distance,
    per_bin_wasserstein_distance,
    mse,
    l1,
)


def test_registry_has_expected_entries():
    for name in [
        "frechet_distance",
        "sliced_wasserstein_distance",
        "histogram_distance",
        "per_bin_wasserstein_distance",
        "mse",
        "l1",
    ]:
        assert name in _dist_dict


def test_mse_zero_for_identical_inputs():
    x = torch.randn(4, 3, 8, 8)
    assert mse(x, x).item() == 0.0


def test_l1_zero_for_identical_inputs():
    x = torch.randn(4, 3, 8, 8)
    assert l1(x, x).item() == 0.0


def test_mse_matches_manual_computation():
    real = torch.randn(3, 5)
    fake = torch.randn(3, 5)
    expected = ((fake - real) ** 2).mean(dim=-1).sum(dim=0)
    assert torch.allclose(mse(real, fake), expected)


def _gaussian_features(n=256, d=4, mean=0.0, std=1.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=g) * std + mean


def test_frechet_distance_identical_distribution_is_near_zero():
    real = _gaussian_features(seed=0)
    fake = _gaussian_features(seed=1)
    assert frechet_distance(real, fake).item() < 0.5


def test_frechet_distance_grows_with_mean_shift():
    real = _gaussian_features(seed=0)
    close = _gaussian_features(seed=1, mean=0.1)
    far = _gaussian_features(seed=1, mean=5.0)
    assert frechet_distance(real, far).item() > frechet_distance(real, close).item()


def test_frechet_distance_nonnegative():
    real = _gaussian_features(seed=0)
    fake = _gaussian_features(seed=1, mean=3.0, std=2.0)
    assert frechet_distance(real, fake).item() >= 0.0


def test_sliced_wasserstein_distance_identical_is_near_zero():
    feats = _gaussian_features(seed=0)
    d = sliced_wasserstein_distance(feats, feats.clone(), nslice=64, bslice=16)
    assert abs(d.item()) < 1e-3


def test_sliced_wasserstein_distance_grows_with_shift():
    real = _gaussian_features(seed=0)
    close = _gaussian_features(seed=1, mean=0.1)
    far = _gaussian_features(seed=1, mean=5.0)
    d_close = sliced_wasserstein_distance(real, close, nslice=64, bslice=16).item()
    d_far = sliced_wasserstein_distance(real, far, nslice=64, bslice=16).item()
    assert d_far > d_close


def test_histogram_distance_identical_is_near_zero():
    feats = _gaussian_features(seed=0)
    d = histogram_distance(feats, feats.clone())
    assert abs(d.item()) < 1e-3


def test_per_bin_wasserstein_distance_shape_and_zero_for_identical():
    feats = _gaussian_features(seed=0, d=5)
    result = per_bin_wasserstein_distance(feats, feats.clone())
    assert set(result.keys()) == {f"bin_{i}" for i in range(5)}
    for v in result.values():
        assert abs(v.item()) < 1e-4

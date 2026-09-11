import torch

from texture_metrics.representations import (
    _repr_dict,
    color_mean,
    color_covariance,
    radial_spectrum,
    cnn_activations,
    spectral_slope,
    spectral_polynomial_embedding,
    RadialProfileExtractor,
    LogRadialProfileExtractor,
)


def test_registry_has_expected_entries():
    for name in [
        "color_mean",
        "color_covariance",
        "radial_spectrum",
        "cnn_activations",
        "cnn_activations_summary",
        "spectral_slope",
        "spectral_polynomial_embedding",
    ]:
        assert name in _repr_dict


def test_color_mean_shape_and_value(target_synth):
    target, _ = target_synth
    out = color_mean(target)
    assert out.shape == (target.shape[0], target.shape[1])
    expected = target.mean(dim=(-2, -1))
    assert torch.allclose(out, expected)


def test_color_covariance_shape(target_synth):
    target, _ = target_synth
    out = color_covariance(target)
    c = target.shape[1]
    n_entries = c * (c + 1) // 2
    assert out.shape == (target.shape[0], n_entries)


def test_radial_spectrum_no_nan(target_synth):
    target, _ = target_synth
    out = radial_spectrum(target)
    assert out.shape[0] == target.shape[0]
    assert not torch.isnan(out).any()


def test_cnn_activations_returns_mean_and_std_per_layer(tiny_cnn, target_synth):
    target, _ = target_synth
    out = cnn_activations(target, tiny_cnn)
    assert set(out.keys()) == {"mean_0", "mean_1", "std_0", "std_1"}
    for v in out.values():
        assert v.shape[0] == target.shape[0]


def test_spectral_slope_finite(target_synth):
    target, _ = target_synth
    img_fft = torch.fft.fft2(target)
    slope = spectral_slope(img_fft, quantile=0.5, keep="high")
    assert torch.isfinite(slope).all()


def test_spectral_polynomial_embedding_shape(target_synth):
    target, _ = target_synth
    img_fft = torch.fft.fft2(target)
    emb = spectral_polynomial_embedding(img_fft, quantile=0.5, polynomial_order=2)
    b, c = target.shape[:2]
    assert emb.shape == (b, c, 3)  # order + 1 coefficients per channel


def test_radial_profile_extractor_modules(target_synth):
    target, _ = target_synth
    rpe = RadialProfileExtractor(bin_size=2)
    out = rpe(target)
    assert out.shape[0] == target.shape[0]
    assert not torch.isnan(out).any()

    log_rpe = LogRadialProfileExtractor(bin_size=2)
    log_out = log_rpe(target)
    assert torch.allclose(log_out, out.log())

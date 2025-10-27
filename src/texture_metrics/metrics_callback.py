from datetime import datetime
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
import time
from typing import Any, Callable, Iterable, List, Optional, Tuple
from functools import wraps
import statistics

import torch
from torch.utils.data import DataLoader
import yaml

from .criteria import fourier
from .criteria import gradients
from .criteria import optimal_transport
from .criteria import style_distances
from .criteria import CNN, CNNOptions, RandomTripletDataset
from .transforms import get_stats, get_transform
from .utils import add_suffixe, format_time, merge_dict, set_seed, stdev


_metric_dict = dict()

def is_valid_metric(metric):
    return metric in _metric_dict

def list_valid_metrics():
    return list(_metric_dict.keys())

def register_metric(func: Callable):
    assert callable(func)
    _metric_dict[func.__name__] = func
    
    return func

# -------------------------------------------------------------------------- #

@dataclass
class MetricsOptions:
    """Texture synthesis metrics options"""
    # Random seed for reproductibility.
    seed:         int  = 0
    # Overwrites the existing metrics.yaml file.
    overwrite:    bool = True
    # Metrics to compute. Full list available in :metrics.py:.
    metrics:      list = field(default_factory=list)
    # Transformation to perform on images befor computing metrics.
    # Follows the same typing as the :transforms: parameter in 
    # :synthesis.py:
    # Use raw: none to perform no transformation.
    transforms:   dict = field(default_factory=dict)
    # Style distance batch size.
    bstyle:       int  = 1
    # Number of slice for SWD on CNN features.
    sstyle:       int  = 1
    # Neural statistics to use : 'mean', 'gram', 'covariance', 'swd'.
    fstyle:       list = field(
        default_factory=lambda: ['mean', 'gram', 'covariance', 'swd'])
    # Projection to use for the :style_distance_projected: metric.
    projections:  dict = field(default_factory=dict)
    # Number of slices for SWD.
    nhist:        int  = 1000
    # SWD batch size.
    bhist:        int  = 250
    # Number of slices for SWD on gradients images.
    ngrad:        int  = 1000
    # Gradients SWD batch size.
    bgrad:        int  = 250
    
    # Model Options to use as a CNN for deep features extraction and style
    # distances computation
    cnn:          CNNOptions = field(default_factory=CNNOptions) 

# ---------------------------------- Style --------------------------------- #

@register_metric
def style_distance(
        target: torch.Tensor, 
        synth: torch.Tensor, 
        cnn: CNN, 
        options: MetricsOptions
    ) -> dict:  
    """Computes 3-channel style distance between target and synthetic
    natural or projected images with CNN.
    Computes style distances using different neural statistics
    (e.g. 'mean', 'gramm', 'covariance', 'swd') given by attribute
    :options.fstyle:.

    Args:
        target (torch.Tensor): target image.
        synth (torch.Tensor): synthetic image.
        cnn (CNN): CNN for neural statistics extraction.
        options (MetricsOptions): metrics options.

    Returns:
        dict: dictionnary containing the style distances.
    """
    target_outputs = cnn(target)
    synth_outputs = cnn(synth)

    results = dict()
    for metric in options.fstyle:
        res = style_distances.weighted_feature_distance(
            synth_outputs, target_outputs, metric, 
            weights=cnn.options.layers_weights, contributions=True, 
            sstyle=options.sstyle
        )
        if res.ndim == 1:
            res.unsqueeze_(0)

        names = [f'{metric}_{i}' for i in range(len(synth_outputs))] \
            + [f'{metric}_total']
        results = {**results, **dict(zip(names, res.T.tolist()))}

    return results

@register_metric
def style_distance_stochastic(
        target: torch.Tensor, 
        synth: torch.Tensor, 
        cnn: CNN, 
        options: MetricsOptions
    ) -> dict:  
    """Computes stochastic style distance between target and
    synthetic multispectral images with CNN.
    Computes style distances using different neural statistics
    (e.g. 'mean', 'gramm', 'covariance', 'swd') given by attribute
    :options.fstyle:.

    Args:
        target (torch.Tensor): target image.
        synth (torch.Tensor): synthetic image.
        cnn (CNN): CNN for neural statistics extraction.
        options (MetricsOptions): metrics options.

    Returns:
        dict: dictionnary containing the style distances.
    """
    # Init random triplet sampler
    random_triplets = RandomTripletDataset(target.shape[-3])
    loader = DataLoader(random_triplets, batch_size=options.bstyle)

    results = torch.zeros(
        len(options.fstyle), 
        len(cnn.options.layers_weights) + 1, 
        device=target.device
    )
    for channels in loader:
        target_outputs = cnn(target[..., channels, :, :].squeeze(0))
        synth_outputs = cnn(synth[..., channels, :, :].squeeze(0))
        for i, metric in enumerate(options.fstyle):
            results[i].add_(
                channels.size(0) 
                * style_distances.weighted_feature_distance(
                    synth_outputs, 
                    target_outputs, 
                    metric, 
                    weights=cnn.options.layers_weights, 
                    contributions=True, 
                    sstyle=options.sstyle
                )
            )
    
    names = []
    for metric in options.fstyle:
        names += [f'metric_{i}' for i in range(len(synth_outputs))] 
        names += ['metric_total']
    
    results.div_(len(random_triplets))
    return dict(zip(names, results.flatten().T.tolist()))

# ------------------------------- Distrbutions ----------------------------- #

def distribution_distances(
        target: torch.Tensor, synth: torch.Tensor, nslice: int):
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
        'swd': optimal_transport.sliced_wasserstein_distance(
            target, synth, nslice=nslice).tolist(),
        **dict(zip(
            [f'band_{i}' for i in range(target.size(-3))],
            optimal_transport.histogram_loss1D(target, synth).sqrt().T.tolist()
        ))
    }

@register_metric
def sliced_wasserstein_distance(
        target: torch.Tensor, synth: torch.Tensor, options: MetricsOptions):
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
        target, synth, nslice=options.nhist, batch_size=options.bhist).tolist()

@register_metric
def histograms(
        target: torch.Tensor, synth: torch.Tensor, options: MetricsOptions):
    """Computes histogram distances.

    Args:
        target (torch.Tensor): target image.
        synth (torch.Tensor): synthetic image.
        options (MetricsOptions): metrics options.

    Returns:
        dict: dictionnary containing histogram distances.
    """
    return distribution_distances(target, synth, nslice=options.bhist)

@register_metric
def color_statistics(
        target: torch.Tensor, synth: torch.Tensor, options: MetricsOptions):
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
        torch.mean((mut - mus)**2, dim=-1) 
        + optimal_transport.bure_distance(covt, covs)
    )
    return {
        'mean': torch.mean((mut - mus)**2, dim=-1).tolist(),
        'covariance': torch.mean((covt - covs)**2, dim=(-1,-2)).tolist(),
        'RX': bure_distance.tolist()
    }

# ----------------------------- Fourier spectra ---------------------------- #

@register_metric
def spectral_radial_distance(
        target: torch.Tensor, synth: torch.Tensor, options: MetricsOptions):
    """Computes L-2 distance on azimuthal spectra (mean and band-wise).

    Args:
        target (torch.Tensor): target image.
        synth (torch.Tensor): synthetic image.
        options (MetricsOptions): metrics options.

    Returns:
        dict: dictionnary containing mean and band-wise radial
            spectral distances.
    """
    names = [f'band_{i}' for i in range(target.size(-3))]
    dist_mean = fourier.spectral_radial_distance(
        target.mean(dim=-3), synth.mean(dim=-3)).sqrt()
    dist_band = fourier.spectral_radial_distance(target, synth).sqrt()
    return {
        'mean': dist_mean.tolist(), 
        **dict(zip(names, dist_band.T.tolist()))
    }

# -------------------------------- Gradients ------------------------------- #

@register_metric
def gradients_distance(
        target: torch.Tensor, synth: torch.Tensor, options: MetricsOptions):
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
        'dx': distribution_distances(dt_x, ds_x, nslice=options.bgrad),
        'dy': distribution_distances(dt_y, ds_y, nslice=options.bgrad),
        'dmag': distribution_distances(dt, ds, nslice=options.bgrad),
    }
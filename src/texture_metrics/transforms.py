from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Callable, Iterable, Optional, Tuple

from matplotlib import pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2

from .constants import MEAN, STD 
from .utils import is_iterable_of_ints, prep_img, spectral_pool

# -------------------------- Color transformation -------------------------- #

def color_operation(func: Callable[[torch.Tensor], torch.Tensor]):
    """Decorator to ease color operation."""
    @wraps(func)
    def wrapper(*args, **kwargs) -> torch.Tensor:
        args = tuple(
            arg.transpose(-3,-1).transpose(-2,0) 
            if isinstance(arg, torch.Tensor) 
            else arg
            for arg in args
        )
        return func(*args, **kwargs).transpose(-2,0).transpose(-3,-1)
    return wrapper

def get_stats(tnsr: torch.Tensor, cholesky=False) \
        -> Tuple[torch.Tensor, torch.Tensor]:
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
    cov = tnsr @ tnsr.transpose(-1,-2) / tnsr.shape[-1]
    if cholesky:
        l = torch.linalg.cholesky(cov)
        return mu.squeeze(-1), l
    else:
        return mu.squeeze(-1), cov

class ColorTransferStats(torch.nn.Module):
    """Color transfer operator. Uses mean and covariance to model
    color distribution."""
    def __init__(self, img1: torch.Tensor,img2: torch.Tensor):
        super().__init__()
        self.mu1, l1 = get_stats(img1, cholesky=True)
        self.mu2, l2 = get_stats(img2, cholesky=True)
        self.t = (l2 @ l1.inverse()).transpose(-1,-2)
        self.t_inv = (l1 @ l2.inverse()).transpose(-1,-2)

    @color_operation
    def forward(self, x: torch.Tensor):
        return ((x - self.mu1).unsqueeze(-2) @ self.t).squeeze(-2) + self.mu2

class NormalizeImagenet(torch.nn.Module):
    """Normalize images to match ImageNet mean and variance."""
    @color_operation
    def forward(self, x: torch.Tensor):
        mu1 = x.mean(dim=(-1,-2))
        s1 = x.std(dim=(-1,-2))
        mu2 = torch.as_tensor(MEAN, dtype=x.dtype, device=x.device)
        s2 = torch.as_tensor(STD, dtype=x.dtype, device=x.device)
        return ((x - mu1) / s1) * s2 + mu2
    
class NormalizeMinMax(torch.nn.Module):
    """Normalize image using minimum and maximum values"""
    @color_operation
    def forward(self, x: torch.Tensor):
        _min, _max = x.min(), x.max()
        return (x - _min) / (_max - _min)

# ------------------------------- Projection ------------------------------- #

class PCATransform(torch.nn.Module):
    """PCA projection operator."""
    def __init__(self, v: torch.Tensor, s: torch.Tensor, mu: torch.Tensor):
        super().__init__()
        self.v = v[:, :3]
        s = s / torch.sum(s ** 2).sqrt()
        self.s = s[:3]
        self.mu = mu

    @color_operation
    def forward(self, x: torch.Tensor):
        return (x - self.mu) @ self.v / self.s
    
class SpectralPool(torch.nn.Module):
    """Performs Spectral Mean Pooling on multispectral images for
    datavisualisation."""
    def __init__(self, nc_out: int):
        super().__init__()
        self.nc_out = nc_out

    def forward(self, x: torch.Tensor):
        return spectral_pool(x, self.nc_out)
    
class BandSelector(torch.nn.Module):
    """Selects bands."""
    def __init__(self, bands: Iterable[int]):
        super().__init__()
        self.bands = bands

    def forward(self, x: torch.Tensor):
        return x[..., self.bands, :, :]

# -------------------------------------------------------------------------- #
    
def get_transform(
        img: torch.Tensor, 
        transform: Optional[str | Path | torch.Tensor | Iterable] = None
    )-> v2.Transform | torch.nn.Module:
    """Gets transform given an image and an instruction.
    Instruction can be :
        - string: 'stochastic', 'none', 'pca', 'norm_imagenet',
            'norm_minmax', 'pooling',
        - Existing transformation saved in .pt or .pth file (provide
        path), 
        - Image to open with PIL (e.g. .png, .jpg) or saved as a
        tensor (e.g. .pt). The image is used as a target for color
        transfer,
        - List of bands to select,
        - Iterable of transforms to perform in sequence.

    Args:
        img (torch.Tensor): input image.
        transform (Optional[str | Path | torch.Tensor | Iterable], optional): 
            transform instruction. 
            Defaults to None.

    Returns:
        v2.Transform | torch.nn.Module: transformation.
    """
    _, c, h, w = img.shape
    if isinstance(transform, torch.Tensor):
        return ColorTransferStats(img, transform).to(img.device)
    elif isinstance(transform, str):
        if transform in ['stochastic', 'none']:
            return v2.Identity().to(img.device)
        elif transform == 'pca':
            a = img.reshape(c, h * w)
            mu = a.mean(dim=1)
            _, s, v = torch.pca_lowrank(a, c, center=True)
            return PCATransform(v, s, mu).to(img.device)
        elif transform == 'norm_imagenet':
            return NormalizeImagenet().to(img.device)
        elif transform == 'norm_minmax':
            return NormalizeMinMax().to(img.device)
        elif transform == 'pooling':
            return SpectralPool(3).to(img.device)
        else:
            return get_transform(img, Path(transform))
    elif isinstance(transform, Path):
        if transform.suffix in ['.pt', '.pth']:
            t = torch.load(
                transform, weights_only=True, map_location=img.device)
            if isinstance(t, (v2.Transform, torch.nn.Module)):
                return t
            elif isinstance(t, torch.Tensor):
                return get_transform(img, t)
            elif isinstance(t, dict):
                try:
                    return PCATransform(
                        *[t[key] for key in ['V', 'lambdas', 'mu']]
                    ).to(img.device)
                except KeyError:
                    pass
        else:
            return get_transform(img, prep_img(transform, device=img.device))
    elif is_iterable_of_ints(transform):
        return BandSelector(transform).to(img.device)
    elif isinstance(transform, Iterable):
        if len(transform) == 1:
            return get_transform(img, transform[0])
        else:
            transforms = get_transform(img, transform[:-1])
            return v2.Compose([
                transforms, 
                get_transform(transforms(img), transform[-1])
            ])
        
    return v2.Identity().to(img.device)
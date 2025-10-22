import torch

from .slicing import sliced_distance

# --------------------------- Transport distances -------------------------- #

def interpolate_histogram(
        x_sorted: torch.Tensor, y_indices: torch.Tensor) -> torch.Tensor:
    """Interpolates on histogram to fit another one.

    Args:
        x_sorted (torch.Tensor): histogram to interpolate.
        y_indices (torch.Tensor): histogram to fit.

    Returns:
        torch.Tensor: interpolated histogram.
    """
    return torch.nn.functional.interpolate(
        input                  = x_sorted, 
        size                   = y_indices.shape[-1],
        mode                   = 'nearest', 
        recompute_scale_factor = False
    )

def histogram_loss1D(x:torch.Tensor, y:torch.Tensor, p:int=2) -> torch.Tensor:
    """Computes $L_p$ distance between sorted histograms of two
    batches of grey-scale images.
    Equivalent to the $p$-Wasserstein distance between the two
    empirical distributions of the images.

    Args:
        x (torch.Tensor): first image.
        y (torch.Tensor): second image.
        p (int, optional): distance order.

    Returns:
        torch.Tensor: $p$-Wasserstein distance at power $p$.
    """
    device = x.device
    xf = x.flatten(start_dim=-2)
    x_sorted, x_indices = torch.sort(xf)
    yf = y.flatten(start_dim=-2)
    y_sorted, y_indices = torch.sort(yf)

    if x_indices.shape[-1] > y_indices.shape[-1]:
        y_sorted = interpolate_histogram(y_sorted, x_indices)
    elif x_indices.shape[-1] < y_indices.shape[-1]:
        x_sorted = interpolate_histogram(x_sorted, y_indices)

    return torch.mean((y_sorted - x_sorted)**p, dim=-1).to(device)

def sliced_wasserstein_distance(
        x:torch.Tensor, y:torch.Tensor, 
        nslice: int, batch_size: int = None, 
        p: int = 2
    ) -> torch.Tensor:
    """Computes the Sliced Wasserstein Distance between the empirical
    distributions of two batches of images.

    1. Projects data onto random directions.
    2. Computes the 1D $p$-Wasserstein distance between the projected
    data.

    Utilizes parallel computing to speed up processing

    Args:
        x (torch.Tensor): first image.
        y (torch.Tensor): second image.
        nslice (int): number of slice (random directions) to use.
        batch_size (int, optional): number of directions to use in
            parallel. 
            If None
            Defaults to None.
        p (int, optional): _description_. Defaults to 2.

    Returns:
        torch.Tensor: _description_
    """
    fn = sliced_distance(histogram_loss1D, nslice, batch_size)
    return fn(x, y, p=p)**(1/p)

def matrix_power(mat: torch.Tensor, p: float) -> torch.Tensor:
    d, u = torch.linalg.eigh(mat)
    d_pos = torch.max(torch.zeros_like(d), d)
    d_pow = torch.diag_embed(torch.pow(d_pos, p))
    return u @ d_pow @ u.mT

def bure_distance(cov1: torch.Tensor, cov2: torch.Tensor)-> torch.Tensor:
    s1 = matrix_power(cov1, .5)
    return torch.einsum(
        'bii->b', 
        cov1 + cov2 - 2 * matrix_power(s1 @ cov2 @ s1, .5)
    )
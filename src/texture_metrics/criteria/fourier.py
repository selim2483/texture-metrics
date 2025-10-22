from typing import Optional, Tuple

import torch

from .slicing import sliced_distance


def periodic_smooth_decomposition(
        u:           torch.Tensor, 
        inverse_dft: bool                     = True, 
        smooth_comp: bool                     = False,
) -> Tuple[torch.Tensor]:
    """Computes periodic + smooth decomposition from Moisan's paper
    (https://link.springer.com/article/10.1007/s10851-010-0227-1).
    When computing discrete Fourier transform, signals are assumed to
    be periodic. The periodic extension of the image then presents
    strong discontinuities since there are no reason for borders to
    be alike. This results in the presence of a cross in the Fourier
    spectrum of the image.
    This function decomposes the image in 2 composent : a periodic
    one and a smooth one. 

    Args:
        u (torch.Tensor): image to process
        inverse_dft (bool, optional): whether to return the inverse
            Fourier transform of the image or not. 
            If True, the function returns the periodic component in
            the image space.
            If False, the functuion returns the Periodic Fourier
            spectrum.
            Defaults to True.

    Returns:
        (torch.Tensor): image or Fourier spectrum periodic + smooth
            decomposition.
    """
    
    u = u.type(torch.complex128)
    # pi = torch.tensor(torch.pi, device=u.device)
    
    arg = 2. * torch.pi * torch.fft.fftfreq(u.shape[-2], 1., device=u.device)
    arg = arg.repeat(*u.shape[:-2], 1, 1).transpose(-2, -1)
    cos_h, sin_h = torch.cos(arg), torch.sin(arg)
    one_minus_exp_h = 1.0 - cos_h - 1j * sin_h

    arg = 2. * torch.pi * torch.fft.fftfreq(u.shape[-1], 1., device=u.device)
    arg = arg.repeat(*u.shape[:-2], 1, 1)
    cos_w, sin_w = torch.cos(arg), torch.sin(arg)
    one_minus_exp_w = 1.0 - cos_w - 1j * sin_w

    w1 = u[..., -1] - u[..., 0]
    w1_dft = torch.fft.fft(w1).unsqueeze(-1)
    v_dft = w1_dft * one_minus_exp_w

    w2 = u[..., -1, :] - u[..., 0, :]
    w2_dft = torch.fft.fft(w2).unsqueeze(-2)
    v_dft = v_dft + one_minus_exp_h * w2_dft

    denom = 2.0 * (cos_h + cos_w - 2.0)
    denom[..., 0, 0] = 1.0

    s_dft = v_dft / denom
    s_dft[..., 0, 0] = 0.0

    if inverse_dft:
        s = torch.fft.ifft2(s_dft).real
        if smooth_comp:
            return u.real - s, s
        else:
            return u.real - s
    else:
        u_dft = torch.fft.fft2(u)
        if smooth_comp:
            return u_dft - s_dft, s_dft
        else:
            return u_dft - s_dft
        
# ---------------------- Orthogonal spectral distance ---------------------- #  
      
def spectral_orthogonal_distance1D(
        x: torch.Tensor, y: torch.Tensor, 
        p: int=2, remove_cross: bool=True
    ) -> torch.Tensor:
    r"""Computes the :math:`L_p` distance in the image space between
    a synthetic image :math:`y` and the set of images having the same
    spectrum (modulus of the 2D Fourier transform) as a reference
    image :math:`x` :

    .. math::
        \mathcal{L}_{spe} (x,y)
        =
        \left\|
        y - \mathcal{F}^{-1} 
        \left(
        \mathcal{F}(y) 
        \times 
        \left\|\frac{\mathcal{F}(x)}{\mathcal{F}(y)}\right\|
        \right)
        \right\|_2^2

    Periodic+smooth decomposition can be performed on images before
    the distance computation by the mean of the :attr:`remove_cross`
    argument.

    The :math:`L_p` space in which the distance is computed can be
    controled by the mean of the :attr:`p` argument.

    Args:
        x (torch.Tensor): reference image.
        y (torch.Tensor): synthetic image.
        p (int, optional): distance order.
        remove_cross (bool, optional): whether to remove the Fourier
            cross or not. 
            Defaults to True.

    Returns:
        torch.Tensor: spectral distance in image space
    """
    if remove_cross :
        x_fft = periodic_smooth_decomposition(x, inverse_dft=False)
        x_fft = torch.fft.fftshift(x_fft, dim=(-2,-1)) 
        y_fft = periodic_smooth_decomposition(y, inverse_dft=False)
        y_fft = torch.fft.fftshift(y_fft, dim=(-2,-1)) 
    else :
        x_fft = torch.fft.fft2(x)
        y_fft = torch.fft.fft2(y)

    # create an grey image with the phase from rec and the module 
    # from the original image
    f_proj = y_fft / (y_fft.abs() + 1e-8) * x_fft.abs()
    proj = torch.fft.ifft2(f_proj).real.detach()
    return torch.norm(y - proj, p=p, dim=(-1, -2))

def sliced_spectral_orthogonal_distance(
        x: torch.Tensor, y: torch.Tensor, 
        nslice: int, batch_size: Optional[int] = None,
        p: int = 2, remove_cross: bool = True
    ):
    fn = sliced_distance(spectral_orthogonal_distance1D, nslice, batch_size)
    return fn(x, y, p=p, remove_cross=remove_cross)

# ------------------------ Radial spectral distance ------------------------ #

def radial_profile(img :torch.Tensor, bin_size :int = 1) -> torch.Tensor:
    """Compute radial profile of the Fourier's transform modulus of a
    grey image.
    Usable for batches of images.

    Args:
        img (torch.Tensor): grey image
        bin_size (int, optional): size of bins used to compute
            histograms.
            Defaults to 1.
            Defaults to "cpu".

    Returns:
        torch.Tensor: Radial profile(s)
    """
    bc = img.shape[:-2]
    img = img.reshape(-1, *img.shape[-2:])
    b, h, w = img.shape 
    fp = periodic_smooth_decomposition(img, inverse_dft=False)
    
    fpshift = torch.fft.fftshift(fp, dim=(-2,-1)) / (h*w)
    mod = fpshift.abs().view(b, -1)**2

    xx, yy = torch.meshgrid(
        torch.arange(fpshift.shape[-2]), 
        torch.arange(fpshift.shape[-1]), 
        indexing='ij'
    )
    r = torch.sqrt((xx - h // 2)**2 + (yy - w // 2)**2)

    # make crowns over which compute the histogram, larger crowns (bin_size>1)
    # should get faster computations
    crowns = (r / bin_size).type(torch.int64).repeat(
        (b, 1)).view(b, -1).to(img.device)

    # compute histogram
    values_sum = torch.zeros(
        b, int(crowns.max() + 1), dtype=mod.dtype, device=img.device)
    values_sum = values_sum.scatter_add_(1, crowns, mod)
    crowns_sum = torch.zeros(
        b, int(crowns.max() + 1), dtype=mod.dtype, device=img.device)
    crowns_sum = crowns_sum.scatter_add_(
        1, crowns, torch.ones(mod.shape, dtype=mod.dtype, device=img.device))
    rad = values_sum / crowns_sum

    return rad.view(*bc, -1)

def spectral_radial_distance(
        x:torch.Tensor, y:torch.Tensor, p:int=2) -> torch.Tensor:
    """Computes Lp distance between the azimuted power spectrum densities
    (PSD) of two grey scale images.

    Args:
        x (torch.Tensor): first image.
        y (torch.Tensor): second image.
        p (int, optional): order of Lp distance. 
            Defaults to 2.

    Returns:
        torch.Tensor: radial PSD $L_p$ distance at power $p$.
    """
    radx = radial_profile(x) 
    rady = radial_profile(y)   
    return torch.mean((torch.log(radx) - torch.log(rady))**p, dim=-1)

def sliced_spectral_radial_distance(
        x:torch.Tensor, y:torch.Tensor, 
        nslice: int, batch_size: Optional[int] = None, 
        p:int=2
    ) -> torch.Tensor:
    fn = sliced_distance(spectral_radial_distance, nslice, batch_size)
    return fn(x, y, p=p)**(1/p)
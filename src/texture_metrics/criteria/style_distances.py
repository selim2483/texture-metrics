from typing import Callable, Iterable

import torch
from torch.nn.functional import mse_loss

from .optimal_transport import sliced_wasserstein_distance


# ----------------------------- Style distance ----------------------------- #

def gramm(tnsr: torch.Tensor | Iterable[torch.Tensor], center_gram=True) -> torch.Tensor:
    """Computes Gram matrix for the input batch tensor.

    Args:
        tnsr (torch.Tensor): input tensor of the Size([B, C, H, W]).
        center_gram (bool, optional): centers feature maps before
            computing gram matrix. 
            Equivalent to covariance matrix. 
            Defaults to True.

    Returns:
        torch.Tensor: output tensor of the Size([B, C, C]).
    """
    if isinstance(tnsr, torch.Tensor):
        b,c,h,w = tnsr.size()
        F = tnsr.view(b, c, h*w)
        if center_gram: 
            F = F - F.mean(dim=-1, keepdim=True)
        G = torch.bmm(F, F.transpose(1,2))
        G.div_(h*w)
        return G
    else:
        return [gramm(t, center_gram=center_gram) for t in tnsr]

def weighted_mse_loss(
        inputs: Iterable[torch.Tensor], 
        targets: Iterable[torch.Tensor], 
        weights: Iterable[float]
    ) -> torch.Tensor:
    """Computes Gram loss.

    Args:
        inputs (Iterable[torch.Tensor]): gram matrices of the inputs.
        targets (Iterable[torch.Tensor]): gram matrices of the
            outputs.
        weights (Iterable[float]): layers weights.

    Returns:
        torch.Tensor: Gram loss.
    """
    losses = torch.empty(size=(len(inputs),))
    for i, (input, target, w) in enumerate(zip(inputs, targets, weights)):
        losses[i] = w * mse_loss(input, target)
    return losses.sum()

def weighted_feature_distance(
        inputs: Iterable[torch.Tensor], 
        targets: Iterable[torch.Tensor], 
        func: Callable | str,
        weights: Iterable[float],
        contributions: bool = False,
        **kwargs
    ) -> torch.Tensor:
    
    distances = torch.empty(
        size=(inputs[0].shape[0], len(inputs)), device=inputs[0].device)
    
    for i, (input, target) in enumerate(zip(inputs, targets)):
        if func == 'mean':
            distances[:, i] = mse_loss(input.mean(), target.mean())
        elif func == 'gramm':
            distances[:, i] = mse_loss(
                gramm(input, center_gram=False), 
                gramm(target, center_gram=False)
            )
        elif func == 'covariance':
            distances[:, i] = mse_loss(gramm(input), gramm(target))
        elif func == 'swd':
            c = input.size(-1)
            print('swd', c, torch.cuda.device_memory_used() / (1024 ** 3))
            distances[:, i] = sliced_wasserstein_distance(
                input, target, 
                nslice=kwargs.get('sstyle') * c, 
                batch_size=c
            )
            print(distances.untyped_storage().nbytes() / (1024 ** 3))
            print('swd', c, torch.cuda.device_memory_used() / (1024 ** 3))
        elif callable(func):
            distances[:, i] = mse_loss(
                func(input, **kwargs), func(target, **kwargs))

    weights_tensor = torch.tensor(weights, device=distances.device)
    total_distance = torch.matmul(distances, weights_tensor)

    if contributions:
        return torch.cat([distances, total_distance.unsqueeze(1)], dim=1)
    else:
        return total_distance
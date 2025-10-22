from functools import wraps
from math import ceil
from typing import Callable, Optional

import torch

def sliced_distance(
        fn: Callable, nslice: int, batch_size: Optional[int]) -> torch.Tensor:
    
    batch_size = batch_size or nslice

    @wraps(fn)
    def wrapper(x: torch.Tensor, y: torch.Tensor, *args, **kwargs):
        res = []
        for i in range(ceil(nslice / batch_size)):
            vs = torch.randn(
                x.shape[-3], batch_size, dtype=x.dtype, device=x.device)
            vs = vs / vs.norm(2, dim=0)

            xs = (x.transpose(-3, -1) @ vs).transpose(-1,-3)
            ys = (y.transpose(-3, -1) @ vs).transpose(-1,-3)

            res.append(fn(xs, ys, *args, **kwargs))
        
        return torch.cat(res).view((i + 1) * batch_size, -1).mean(dim=0)
    
    return wrapper
import torch


def image_gradient(im:torch.Tensor, result: str = 'all'):
    dx, dy = im.diff(dim=-2)[...,:-1], im.diff(dim=-1)[...,:-1,:]
    mag = torch.sqrt(dx**2 + dy**2)
    
    if result == 'all':
        return dx, dy, mag
    elif result == 'mag':
        return mag
    elif result == 'dx':
        return dx
    elif result == 'dy':
        return dy
    elif result == 'dx_dy':
        return dx, dy
from math import ceil
from pathlib import Path
import random
import statistics
from typing import Iterable, Optional, Union

import numpy as np
import torch
from torchvision.transforms.functional import center_crop, resize, to_tensor
from PIL import Image

from .constants import MEAN

# ---------------------- Image processing and logging ---------------------- #

def prep_img(
        image: Path | str | torch.Tensor, 
        size: int = None,
        source: Optional[str] = None,
        bands: Optional[list] = None, 
        device: torch.device = 'cpu'
    ):
    """Preprocess image.
    1) (load as PIL or Tensor),
    2) resize,
    3) (convert to tensor),
    4) remove alpha channel if any
    """
    if isinstance(image, torch.Tensor):
        tensor = resize(image, size or image.shape[-2:])
        if tensor.ndim == 3:
            return tensor.unsqueeze(0)
        elif tensor.ndim == 4:
            return tensor
    elif isinstance(image, str):
        return prep_img(Path(image), size, source, bands, device)
    elif isinstance(image, Path):
        if image.suffix == '.pt':
            tensor = torch.load(image, weights_only=True, map_location=device)
            if source == 'hytexila':
                tensor = 2 * center_crop(tensor.transpose(0,2), 512) - 1
            if bands != None:
                tensor = tensor[bands]
        else:
            im = Image.open(image)
            tensor = to_tensor(im).to(device) * 2 - 1
            if tensor.shape[-3]==4:
                print('removing alpha chanel')
                tensor = tensor[...,:3,:,:]
        return prep_img(tensor, size, source, bands, device)

def normalize(tensor: torch.Tensor) -> torch.Tensor:
    """Normalize image with ImageNet statistics.
    (-1, 1) -> (0, 1) -> (-127.5, 127.5)

    Args:
        tensor (torch.Tensor): image.

    Returns:
        torch.Tensor: normalized image.
    """
    mean = torch.as_tensor(
        MEAN if tensor.shape[1]==3 else [.5 for _ in range(tensor.shape[1])], 
        dtype=tensor.dtype, device=tensor.device
    ).view(-1, 1, 1)
    tensor = tensor / 2 + .5
    tensor.sub_(mean).mul_(255)
    return tensor

def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Denormalize image into (0, 1) range.

    Args:
        tensor (torch.Tensor): image

    Returns:
        torch.Tensor: denormalized image.
    """
    tensor = tensor.clone().squeeze() 
    tensor = tensor / 2 + .5
    return tensor

def spectral_pool(x: torch.Tensor, nc_out: int = 3) -> torch.Tensor:
    """Perform mean pooling along spectral dimension.

    Args:
        x (torch.Tensor): image
        nc_out (int, optional): number of channels to output. 
            Defaults to 3.

    Returns:
        torch.Tensor: pooled image.
    """
    b, c, w, h = x.size()
    kernel_size = ceil(c / nc_out)
    if c==1 :
        return x.repeat((1, 3, 1, 1))
    else :
        x = x.reshape(b, c, h * w).permute(0, 2, 1)
        pooled = torch.nn.functional.avg_pool1d(
            x, kernel_size=kernel_size, ceil_mode=True)
        return pooled.permute(0, 2, 1).view(b, nc_out, w, h)

def to_logger(tensor: torch.Tensor):
    """Converts tensor to log in Tensorboard.

    Args:
        tensor (torch.Tensor): image.

    Returns:
        torch.Tensor: denormalized and cliped image.
    """
    img = spectral_pool(tensor).clone().detach().cpu()
    img = denormalize(img)
    img = img.clip(0, 1)
    return img

def remove_files(directory_path: str | Path):
    """Removes file in directory

    Args:
        directory_path (str | Path): path of directory.
    """
    if isinstance(directory_path, str):
        return remove_files(Path(directory_path))
    elif isinstance(directory_path, Path):    
        if not directory_path.exists() or not directory_path.is_dir():
            return
        
        for item in directory_path.iterdir():
            if item.is_file():
                item.unlink()

    print(f"All files from {directory_path} have been removed")

# ----------------------------- Metrics logging ---------------------------- #

def format_time(seconds: Union[int, float]) -> str:
    """Convert the seconds to human readable string with days, hours,
    minutes and seconds."""
    s = int(np.rint(seconds))

    if s < 60:
        return "{0}s".format(s)
    elif s < 60 * 60:
        return "{0}m {1:02}s".format(s // 60, s % 60)
    elif s < 24 * 60 * 60:
        return "{0}h {1:02}m {2:02}s".format(
            s // (60 * 60), (s // 60) % 60, s % 60)
    else:
        return "{0}d {1:02}h {2:02}m".format(
            s // (24 * 60 * 60), (s // (60 * 60)) % 24, (s // 60) % 60)

def merge_dict(dict1: dict, dict2: dict):
    """Merges two dict with priority to the second in case of
    conflicts."""
    if isinstance(dict1, dict):
        for k,v in dict2.items():
            if isinstance(v, dict):
                dict1.update({k:merge_dict(dict1.get(k, {}), v)})
            else:
                dict1.update({k:v})
        return dict1
    else:
        return dict2
    
def add_suffixe(name: str, suffixe: str | None):
    """Add suffix to string for logging.
    If raw or None, no suffixe is added."""
    if suffixe == None or suffixe == 'raw':
        return name
    else:
        return f'{name}_{suffixe}'

# ---------------------------------- Misc ---------------------------------- #

def is_iterable_of_ints(obj):
    """Check if object is an Iterable of int."""
    return isinstance(obj, Iterable) and all(isinstance(x, int) for x in obj)

def set_seed(seed: int = 42):
    """Fixes all random seed to unsure reproductibility.

    Args:
        seed (int, optional): Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 

    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False  

# ---------------------------------- Stats ---------------------------------- #

def stdev(data: list|tuple):
    if len(data) == 1:
        return 0.
    else:
        return statistics.stdev(data)
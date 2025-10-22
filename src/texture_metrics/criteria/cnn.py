from dataclasses import dataclass
import itertools
import numpy as np
import torchvision.models as models
from pathlib import Path
from typing import Iterable, List, Optional, Union

import torch
from torch.utils.data import Dataset

from .style_distances import gramm
from ..constants import LAYERS
from ..utils import normalize

@dataclass
class CNNOptions:
    """Model options of the CNN"""
    # Architecture of the CNN (vgg[11,13,16,19], resnet[34,50], 
    # googlenet)
    architecture: str = 'vgg19'
    # Weights of the CNN. 
    # If None, use random initialization. 
    # If 'imagenet', use ImageNet default pretrained weights. 
    # Else, should be the path of the pickle containing the
    # pretrained weights.
    weights: str = 'imagenet'
    # Layers to extract for Gram matrices
    layers: Optional[List[int]] = None
    # Weights $w_l$ of the quadratic differences between Gram matrices 
    layers_weights: Optional[List[float]] = None
    
    def __post_init__(self):
        self.layers = self.layers or LAYERS[self.architecture][0]
        self.layers_weights = (
            self.layers_weights or LAYERS[self.architecture][1])
        
# ---------------------------------- Model --------------------------------- #

def get_module_recursive(
    module: torch.nn.Module, layer: Union[str, Iterable]):
    if isinstance(layer, str):
        return module._modules[layer]
    elif isinstance(layer, int):
        return module[layer]
    elif len(layer) == 1:
        return get_module_recursive(module, layer[0])
    else:
        return get_module_recursive(
            get_module_recursive(module, layer[0]), layer[1:])
    
class CNN(torch.nn.Module):
    """Wrapper class of CNN for texture synthesis."""
    def __init__(self, options: CNNOptions):
        super().__init__()
        self.options = options
        weights = 'DEFAULT' if self.options.weights == 'imagenet' else None
        if self.options.architecture == 'vgg11':
            self.model = models.vgg11(weights=weights).features
        elif self.options.architecture == 'vgg13':
            self.model = models.vgg13(weights=weights).features
        elif self.options.architecture == 'vgg16':
            self.model = models.vgg16(weights=weights).features
        elif self.options.architecture == 'vgg19':
            self.model = models.vgg19(weights=weights).features
            if Path(self.options.weights).is_file():
                self.model = models.vgg19().features
                try:
                    pretrained_dict = torch.load(
                        self.options.weights)["state_dict_extractor"]
                    self.model.load_state_dict(pretrained_dict)
                except:
                    pretrained_dict = torch.load(self.options.weights)
                    for param, item in zip(
                        self.model.parameters(), pretrained_dict.keys()):
                        param.data = pretrained_dict[item].type(
                            torch.FloatTensor)
        elif self.options.architecture == 'resnet34':
            self.model = models.resnet34(weights=weights)
        elif self.options.architecture == 'resnet50':
            self.model = models.resnet50(weights=weights)
        elif self.options.architecture == 'googlenet':
            self.model = models.googlenet(weights=weights)

        self.model.requires_grad_(False)
        self.model.eval()

        # Hook definition
        self.outputs = {}

        def save_output(name):

            # The hook signature
            def hook(module, module_in, module_out):
                self.outputs[name] = module_out
            return hook

        # Register hook on each layer with index on array "layers"
        for layer in self.options.layers:
            handle = get_module_recursive(
                self.model, layer).register_forward_hook(save_output(layer))
        
    def forward(self, x: torch.Tensor):
        self.model(normalize(x))
        return [self.outputs[key] for key in self.options.layers]
    
# ----------------------------- Random triplet ----------------------------- #

class RandomTripletDataset(Dataset):
    """Dataset to sample random triplets."""

    def __init__(self, nchannels: int):
        """Creates a `RandomTripletDataset`.

        Args:
            nchannels (int): number of channels.
        """
        self.channels = list(itertools.permutations(range(nchannels), 3))

    def __getitem__(self, index):
        return torch.tensor(self.channels[index])
    
    def __len__(self):
        return len(self.channels)
    
class InfiniteSampler(torch.utils.data.Sampler):
    def __init__(
            self, 
            dataset, 
            rank         = 0, 
            num_replicas = 1, 
            shuffle      = True, 
            seed         = 0, 
            window_size  = 0.5
        ):
        assert len(dataset) > 0
        assert num_replicas > 0
        assert 0 <= rank < num_replicas
        assert 0 <= window_size <= 1
        super().__init__()
        self.dataset = dataset
        self.rank = rank
        self.num_replicas = num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.window_size = window_size

    def __iter__(self):
        order = np.arange(len(self.dataset))
        rnd = None
        window = 0
        if self.shuffle:
            rnd = np.random.RandomState(self.seed)
            rnd.shuffle(order)
            window = int(np.rint(order.size * self.window_size))

        idx = 0
        while True:
            i = idx % order.size
            if idx % self.num_replicas==self.rank:
                yield order[i]
            if window >= 2:
                j = (i - rnd.randint(window)) % order.size
                order[i], order[j] = order[j], order[i]
            idx += 1
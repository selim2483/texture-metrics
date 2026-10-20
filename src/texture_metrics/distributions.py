import math
import time
from typing import Any, Callable, Optional, Union

import torch
from torchmetrics import Metric
from torchmetrics.image.fid import FrechetInceptionDistance, NoTrainInceptionV3
from torchmetrics.utilities import dim_zero_cat

from .criteria.cnn import CNN

from .representations import _repr_dict, cnn_activations
from .distances import _dist_dict, monge_inception_distance


class DistributionDistance(Metric):
    """Base class for distribution metrics.

    Args:
        repr_fn (Callable | str): function to compute the image representation.
        repr_fn_kwargs (dict): keyword arguments for the representation function.
        dist_fn (Callable | str): function to compute the distance between two distributions.
        dist_fn_kwargs (dict): keyword arguments for the distance function.
    """

    def __init__(
        self,
        repr_fn: Callable | str | torch.nn.Module,
        dist_fn: Callable | str,
        repr_fn_kwargs: dict = {},
        dist_fn_kwargs: dict = {},
        max_real_samples: int | float = math.inf,
        name: Optional[str] = None,
    ):
        super().__init__()

        if isinstance(repr_fn, str):
            repr_fn = _repr_dict[repr_fn]
        self.repr_fn = repr_fn
        self.repr_fn_kwargs = repr_fn_kwargs or {}

        if isinstance(dist_fn, str):
            dist_fn = _dist_dict[dist_fn]
        self.dist_fn = dist_fn
        self.dist_fn_kwargs = dist_fn_kwargs or {}

        self.add_state("real", default=[], dist_reduce_fx="cat")
        self.add_state("fake", default=[], dist_reduce_fx="cat")
        self.add_state("time", default=torch.tensor(0.0), dist_reduce_fx="sum")

        self.max_real_samples = max_real_samples

        if name is None:
            name = f"{repr_fn.__name__}/{dist_fn.__name__}"
        self.name = name

    def update(self, imgs: torch.Tensor, targets: torch.Tensor):
        """Update the metric state with a batch of images and their targets.

        Args:
            imgs (torch.Tensor): batch of images, shape (B, C, H, W).
            targets (torch.Tensor): batch of target images, shape (B, C, H, W).
        """
        self._update(imgs, real=False)
        self._update(targets, real=True)

    def _update(self, imgs: torch.Tensor, real: bool):
        """Update the metric state with a batch of images.

        Unlike FID, the underlying 1D-Wasserstein distance falls back
        to nearest-neighbor interpolation when the real/fake sample
        counts differ, which degrades badly for large mismatches — so
        ``max_real_samples`` is applied symmetrically here: fake
        samples are capped at the same count as real ones, never
        allowed to outnumber them.

        Args:
            imgs (torch.Tensor): batch of images, shape (B, C, H, W).
            real (bool): whether the images are real or fake.
        """
        start_time = time.time()
        reprs = self.repr_fn(imgs, **self.repr_fn_kwargs).reshape(
            imgs.shape[0], -1
        )
        target = self.real if real else self.fake
        num_samples = sum(t.shape[0] for t in target)
        if num_samples < self.max_real_samples:
            target.append(reprs)
        self.time = self.time + (time.time() - start_time)

    def compute(self) -> torch.Tensor:
        """Compute the metric value.

        Returns:
            torch.Tensor: scalar metric value.
        """
        start_time = time.time()
        reals = dim_zero_cat(self.real)
        target = dim_zero_cat(self.fake)
        result = self.dist_fn(reals, target, **self.dist_fn_kwargs)
        self.time = self.time + (time.time() - start_time)
        return result


class CNNDistributionDistance(Metric):
    """Distribution metric that uses a CNN to compute the image representation.

    Args:
        cnn (torch.nn.Module | dict): CNN model (or constructor kwargs
            for ``texture_metrics.criteria.cnn.CNN``) to compute the
            image representation.
        repr_fn (Callable | str): function to compute the image representation.
        repr_fn_kwargs (dict): keyword arguments for the representation function.
        dist_fn (Callable | str): function to compute the distance between two distributions.
        dist_fn_kwargs (dict): keyword arguments for the distance function.
    """

    def __init__(
        self,
        cnn: Optional[CNN | dict] = None,
        dist_fn: Callable | str = "swd",
        repr_fn_kwargs: dict = {},
        dist_fn_kwargs: dict = {},
        max_real_samples: int | float = math.inf,
        name: Optional[str] = None,
        input_image_size: tuple[int, int, int] = (3, 64, 64),
        compile: bool = True,
    ):
        super().__init__()

        if isinstance(cnn, dict):
            cnn = CNN(**cnn)
        elif cnn is None:
            cnn = CNN()
        self.cnn = cnn
        if compile:
            self.cnn.compile()
        self.repr_fn_kwargs = repr_fn_kwargs or {}

        if isinstance(dist_fn, str):
            dist_fn = _dist_dict[dist_fn]
        self.dist_fn = dist_fn
        self.dist_fn_kwargs = dist_fn_kwargs or {}

        self.input_image_size = input_image_size
        dummy_tensor = torch.randn(1, *self.input_image_size)
        dummy_output = cnn_activations(
            dummy_tensor, self.cnn, **self.repr_fn_kwargs
        )
        self.keys = list(dummy_output.keys())
        for key in self.keys:
            self.add_state(f"real_{key}", default=[], dist_reduce_fx="cat")
            self.add_state(f"fake_{key}", default=[], dist_reduce_fx="cat")
        self.add_state("time", default=torch.tensor(0.0), dist_reduce_fx="sum")

        self.max_real_samples = max_real_samples

        if name is None:
            name = f"cnn_activations/{dist_fn.__name__}"
        self.name = name

    def update(self, imgs: torch.Tensor, targets: torch.Tensor):
        """Update the metric state with a batch of images and their targets.

        Args:
            imgs (torch.Tensor): batch of images, shape (B, C, H, W).
            targets (torch.Tensor): batch of target images, shape (B, C, H, W).
        """
        self._update(imgs, real=False)
        self._update(targets, real=True)

    def _update(self, imgs: torch.Tensor, real: bool):
        """Update the metric state with a batch of images.

        Unlike FID, the underlying 1D-Wasserstein distance falls back
        to nearest-neighbor interpolation when the real/fake sample
        counts differ, which degrades badly for large mismatches — so
        ``max_real_samples`` is applied symmetrically here: fake
        samples are capped at the same count as real ones, never
        allowed to outnumber them.

        Args:
            imgs (torch.Tensor): batch of images, shape (B, C, H, W).
            real (bool): whether the images are real or fake.
        """
        start_time = time.time()
        reprs = cnn_activations(imgs, self.cnn, **self.repr_fn_kwargs)
        prefix = "real" if real else "fake"
        for key, value in reprs.items():
            target = getattr(self, f"{prefix}_{key}")
            num_samples = sum(t.shape[0] for t in target)
            if num_samples < self.max_real_samples:
                target.append(value.reshape(imgs.shape[0], -1))
        self.time = self.time + (time.time() - start_time)

    def compute(self) -> dict[str, torch.Tensor]:
        """Compute the metric value.

        Returns:
            dict[str, torch.Tensor]: dictionary of metric values for each representation.
        """
        start_time = time.time()
        metrics = {}
        for key in self.keys:
            reals = dim_zero_cat(getattr(self, f"real_{key}"))
            target = dim_zero_cat(getattr(self, f"fake_{key}"))
            metrics[key] = self.dist_fn(reals, target, **self.dist_fn_kwargs)
        self.time = self.time + (time.time() - start_time)
        return metrics


class FID(FrechetInceptionDistance):
    def __init__(
        self,
        feature: Union[int, torch.nn.Module] = 2048,
        reset_real_features: bool = True,
        input_img_size: tuple[int, int, int] = (3, 299, 299),
        feature_extractor_weights_path: Optional[str] = None,
        antialias: bool = True,
        value_range: tuple[float, float] = (0.0, 1.0),
        max_real_samples: int | float = math.inf,
        name: str = "fid",
        compile: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            feature=feature,
            reset_real_features=reset_real_features,
            normalize=True,
            input_img_size=input_img_size,
            feature_extractor_weights_path=feature_extractor_weights_path,
            antialias=antialias,
            **kwargs,
        )
        if compile:
            self.inception.compile()
        self.vmin, self.vmax = value_range
        self.max_real_samples = max_real_samples
        self.name = name
        self.add_state("time", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, imgs: torch.Tensor, targets: torch.Tensor):
        """Update the metric state with a batch of images and their targets.

        Args:
            imgs (torch.Tensor): batch of images, shape (B, C, H, W).
            targets (torch.Tensor): batch of target images, shape (B, C, H, W).
        """
        self._update(imgs, real=False)
        self._update(targets, real=True)

    def _update(self, imgs: torch.Tensor, real: bool):
        if real and self.real_features_num_samples >= self.max_real_samples:
            return

        start_time = time.time()
        imgs = (imgs - self.vmin) / (self.vmax - self.vmin)
        imgs = torch.clamp(imgs, 0.0, 1.0)
        super().update(imgs, real)
        self.time = self.time + (time.time() - start_time)

    def compute(self) -> torch.Tensor:
        """Compute the metric value.

        Returns:
            torch.Tensor: scalar FID value.
        """
        start_time = time.time()
        result = super().compute()
        self.time = self.time + (time.time() - start_time)
        return result


class MIND(Metric):
    """Monge Inception Distance.

    Stores the features of real and fake images and compares them with
    :func:`texture_metrics.distances.monge_inception_distance`. Unlike
    FID, MIND is reliable with few samples, which makes it cheap enough
    to track during training.

    Args:
        feature (int | torch.nn.Module): either an InceptionV3 feature
            layer (64, 192, 768 or 2048) as for FID, or a custom feature
            extractor mapping (B, C, H, W) images in [0, 1] to (B, D)
            features.
        reset_real_features (bool): whether ``reset`` also clears the
            real features. Set to False to extract them only once.
        feature_extractor_weights_path (str, optional): path to the
            InceptionV3 weights.
        antialias (bool): whether InceptionV3 uses antialiased resizing.
        value_range (tuple[float, float]): range of the input images,
            rescaled to [0, 1] before feature extraction.
        max_real_samples (int | float): maximum number of stored samples,
            applied to both real and fake features.
        n_projections (int): number of random projections.
        projection_batch_size (int, optional): number of projections
            processed at once, to bound memory with many samples. None
            processes them all at once.
        rng_seed (int): seed of the random projections.
        name (str): metric name.
        compile (bool): whether to compile the feature extractor.
    """

    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(
        self,
        feature: Union[int, torch.nn.Module] = 2048,
        reset_real_features: bool = True,
        feature_extractor_weights_path: Optional[str] = None,
        antialias: bool = True,
        value_range: tuple[float, float] = (0.0, 1.0),
        max_real_samples: int | float = math.inf,
        n_projections: int = 100,
        projection_batch_size: Optional[int] = None,
        rng_seed: int = 0,
        name: str = "mind",
        compile: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if isinstance(feature, int):
            valid_int_input = (64, 192, 768, 2048)
            if feature not in valid_int_input:
                raise ValueError(
                    f"Integer input to argument `feature` must be one of "
                    f"{valid_int_input}, but got {feature}."
                )
            self.feature_extractor = NoTrainInceptionV3(
                name="inception-v3-compat",
                features_list=[str(feature)],
                feature_extractor_weights_path=feature_extractor_weights_path,
                antialias=antialias,
            )
            self.used_custom_model = False
        elif isinstance(feature, torch.nn.Module):
            self.feature_extractor = feature
            self.used_custom_model = True
        else:
            raise TypeError("Got unknown input to argument `feature`")
        if compile:
            self.feature_extractor.compile()

        self.reset_real_features = reset_real_features
        self.vmin, self.vmax = value_range
        self.max_real_samples = max_real_samples
        self.n_projections = n_projections
        self.projection_batch_size = projection_batch_size
        self.rng_seed = rng_seed
        self.name = name

        self.add_state("real", default=[], dist_reduce_fx="cat")
        self.add_state("fake", default=[], dist_reduce_fx="cat")
        self.add_state("time", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, imgs: torch.Tensor, targets: torch.Tensor):
        """Update the metric state with a batch of images and their targets.

        Args:
            imgs (torch.Tensor): batch of images, shape (B, C, H, W).
            targets (torch.Tensor): batch of target images, shape (B, C, H, W).
        """
        self._update(imgs, real=False)
        self._update(targets, real=True)

    def _update(self, imgs: torch.Tensor, real: bool):
        """Update the metric state with a batch of images.

        Args:
            imgs (torch.Tensor): batch of images, shape (B, C, H, W).
            real (bool): whether the images are real or fake.
        """
        target = self.real if real else self.fake
        if sum(t.shape[0] for t in target) >= self.max_real_samples:
            return

        start_time = time.time()
        imgs = (imgs - self.vmin) / (self.vmax - self.vmin)
        imgs = torch.clamp(imgs, 0.0, 1.0)
        if not self.used_custom_model:
            imgs = (imgs * 255).byte()
        features = self.feature_extractor(imgs).reshape(imgs.shape[0], -1)
        target.append(features.double())
        self.time = self.time + (time.time() - start_time)

    def compute(self) -> torch.Tensor:
        """Compute the metric value.

        Real and fake features are truncated to their common number of
        samples, as MIND requires equally sized sets.

        Returns:
            torch.Tensor: scalar MIND value.
        """
        start_time = time.time()
        reals = dim_zero_cat(self.real)
        fakes = dim_zero_cat(self.fake)
        n = min(reals.shape[0], fakes.shape[0])
        result = monge_inception_distance(
            reals[:n],
            fakes[:n],
            rng_seed=self.rng_seed,
            n_projections=self.n_projections,
            projection_batch_size=self.projection_batch_size,
        ).float()
        self.time = self.time + (time.time() - start_time)
        return result

    def reset(self) -> None:
        """Reset the metric state, keeping real features if
        ``reset_real_features`` is False."""
        if self.reset_real_features:
            super().reset()
            return
        # states missing from _defaults are left untouched by reset
        default = self._defaults.pop("real")
        super().reset()
        self._defaults["real"] = default

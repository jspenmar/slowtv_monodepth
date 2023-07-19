import random
from abc import abstractmethod

import kornia.augmentation as ka
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

import src.typing as ty
from src.tools import ops, viz
from src.utils import io
from . import BaseDataset

__all__ = ['MdeBaseDataset']


class MdeBaseDataset(BaseDataset, retry_exc=ty.SuppImageNotFoundError):
    """Base class used for Monocular Depth Estimation datasets.
    See the documentation from `BaseDataset` for additional information.

    Assumes most datasets provide:
        - Image: Target image from which to predict depth.
        - Support: Adjacent frames (either monocular or stereo) used to compute photometric consistency losses.
        - Depth: Target ground-truth depth.
        - K: Camera intrinsic parameters.

    Datatypes can be added/removed by child classes as required by modifying class attribute `VALID_DATUM`.
    All child classes must provide a class attribute SHAPE, containing the original image resolution as (H, W).

    In general, the functions `load_<datatype>` load the corresponding item in the dataset and store it in the required
    dictionary with the desired key, e.g. images are added to `(x, y)`, while ground-truth depth is only added to `y`.
    To allow for slightly more modular loading, the actual interacting with the dataset devkit is further isolated to
    the `_load_<datatype>` functions. These are the ones that subclasses must implement.

    Parameters:
    :param shape: (int, int) Target image shape as (h, w).
    :param datum: (list[str]) List of datatypes to load.
    :param supp_idxs: (None|int|list[int]) Support frame indexes relative to the target frame.
    :param randomize_supp: (bool) If `True` randomize the support index for each item. (Same for all support frames)

    Attributes:
    :attr h: (int) Image height.
    :attr w: (int) Image width.
    :attr size: (int) Image size as (w, h).
    :attr SHAPE: (REQUIRED) (int, int) Original image shape as (h, w).
    :attr H: (int) Original image width.
    :attr W: (int) Original image width.
    :attr SIZE: (int) Original image size as (w, h).
    :attr prob_flip: (float) Probability to apply horizontal flipping augmentation.
    :attr prob_photo: (float) Probability to apply photometric jittering augmentation.

    Methods:
    :method load_image: Load target image and keep un-augmented copy. (x, y)
    :method _load_image: (REQUIRED) Load target image from dataset.
    :method load_support: Load all support frames (including stereo) and keep un-augmented copy. (x, y)
    :method _get_supp_scale: Generate the index of the support frame relative to the target image.
    :method _load_stereo_image: (REQUIRED) Load the support stereo frame from dataset.
    :method _load_stereo_T: (REQUIRED) Load the stereo transform to the stereo frame from dataset.
    :method load_depth: Load ground-truth depth and store in loss targets. (y)
    :method _load_depth: (REQUIRED) Load ground-truth depth from dataset.
    :method load_K: Load camera intrinsics and store in loss targets. (y)
    :method _load_K: (REQUIRED) Load camera intrinsics from dataset.
    :method apply_flip_aug: Apply horizontal flipping augmentation.
    :method apply_photo_aug: Apply colour jittering augmentation to `x`.
    """
    VALID_DATUM = 'image support depth K'

    def __init__(self,
                 shape: tuple[int, int] = None,
                 datum: ty.U[str, ty.S[str]] = 'image K',
                 supp_idxs: ty.N[ty.U[int, ty.S[int]]] = None,
                 randomize_supp: bool = False,
                 augmentations=None,
                 **kwargs):
        super().__init__(datum=datum, **kwargs)
        self.shape = shape or self.SHAPE
        self.supp_idxs = supp_idxs or []
        self.randomize_supp = randomize_supp
        self.should_resize = shape is not None
        self.augmentations = augmentations or {}

        if isinstance(self.supp_idxs, int): self.supp_idxs = [self.supp_idxs]

        # Augmentations
        self.prob_flip = self.augmentations.get('flip', 0 if self.augmentations else 0.5)
        self.prob_photo = self.augmentations.get('photo', 0 if self.augmentations else 0.5)

        self.photo = ka.ColorJiggle(
            brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2), hue=(-0.1, 0.1),
            p=1.0, same_on_batch=True, keepdim=True,
        )
        self.flip = lambda arr, axis=1: np.ascontiguousarray(np.flip(arr, axis=axis))

    @property
    def h(self) -> int:
        """Image height."""
        return self.shape[0]

    @property
    def w(self) -> int:
        """Image width."""
        return self.shape[1]

    @property
    def size(self) -> tuple[int, int]:
        """Image size as (w, h)."""
        return self.w, self.h

    @property
    @abstractmethod
    def SHAPE(self) -> tuple[int, int]:
        """Original image shape as (H, W)."""

    @property
    def H(self) -> int:
        """Original image height."""
        return self.SHAPE[0]

    @property
    def W(self) -> int:
        """Original image width."""
        return self.SHAPE[1]

    @property
    def SIZE(self) -> tuple[int, int]:
        """Original image size as (W, H)."""
        return self.W, self.H

    def log_args(self) -> None:
        """Log creation arguments."""
        self.logger.info(f"Loading images with shape '{self.shape}'...")
        if self.supp_idxs: self.logger.debug(f"Logging '{self.supp_idxs}' support frames...")
        if self.randomize_supp: self.logger.debug(f"Randomizing index of support frames...")
        super().log_args()

    def validate_args(self) -> None:
        """Error checking for provided dataset configuration."""
        super().validate_args()

        if self.h > self.w:
            self.logger.warning(f'Image height={self.h} is greater than image width={self.w}. '
                                f'Did you pass these in the correct order? Expected (height, width).')

        if self.H > self.W:
            self.logger.warning(f'Image full height={self.H} is greater than image width={self.W}. '
                                f'Did you pass these in the correct order? Expected SHAPE as (Height, Width).')

        for i in self.supp_idxs:
            if self.randomize_supp and abs(i) not in {0, 1}:
                raise ValueError(f'Invalid supplementary index when randomizing. ({i} vs. {{+1, 0, -1}} )')

        if self.supp_idxs and not self.has('support'):
            raise ValueError('Support indexes were provided, but `support` was not found in `datum`.')

        if self.has('support') and not self.supp_idxs:
            raise ValueError('Support images were requested, but no indexes were provided.')

    @classmethod
    def collate_fn(cls, batch: ty.S[ty.BatchData]) -> ty.BatchData:
        """Classmethod to collate multiple dataset items into a batch.

        `x['supp_idxs']` is converted into a single Tensor, since indexes are the same for all items. (n, )
        `x['supp_imgs']` are transposed to reflect the number of support images as the first dimension. (n, b, ...)
        `y['supp_imgs']` as above. (n, b, ...)

        :param batch: (ty.BatchData) List of dataset items, each with `(x, y, m)`.
        :return: (Dataset) Collated batch, where all items are stacked into a single tensor.
        """
        x, y, m = super().collate_fn(batch)

        if 'supp_idxs' in x:
            x['supp_idxs'] = x['supp_idxs'][0]  # Keep a single list of support idxs
            x['supp_imgs'] = x['supp_imgs'].transpose(0, 1)  # (n, b, 3, h, w)
            y['supp_imgs'] = y['supp_imgs'].transpose(0, 1)

        return x, y, m

    def load_image(self, data: ty.Any, batch: ty.BatchData) -> ty.BatchData:
        """Load target image and keep un-augmented copy. (x, y)"""
        x, y, m = batch
        img = self._load_image(data)
        x['imgs'] = io.pil2np(img)
        y['imgs'] = x['imgs'].copy()  # Non-augmented copy
        return batch

    @abstractmethod
    def _load_image(self, data: ty.Any, offset: int = 0) -> Image:
        """Load target image from dataset. Offset should be used when loading support frames."""

    def load_support(self, data: ty.Any, batch: ty.BatchData) -> ty.BatchData:
        """Load all support frames (including stereo) and keep un-augmented copy. (x, y)"""
        x, y, m = batch
        x['supp_idxs'] = np.array(self.supp_idxs)

        supp, k = [], self.get_supp_scale(data)
        m['supp'] = str(k)
        for i in self.supp_idxs:
            i *= k
            self.logger.debug(f'Loading support image: {i}')
            if i == 0:
                supp.append(self._load_stereo_image(data))
                y['T_stereo'] = self._load_stereo_T(data)
            else:
                supp.append(self._load_image(data, offset=i))

        x['supp_imgs'] = np.stack([io.pil2np(img) for img in supp])
        y['supp_imgs'] = x['supp_imgs'].copy()
        return batch

    def get_supp_scale(self, data: ty.Any) -> int:
        """Generate the index of the support frame relative to the target image."""
        return 1

    @abstractmethod
    def _load_stereo_image(self, data: ty.Any) -> Image:
        """Load the support stereo frame from dataset."""

    @abstractmethod
    def _load_stereo_T(self, data: ty.Any) -> ty.A:
        """Load the stereo transform to the stereo frame from dataset."""

    def load_depth(self, data: ty.Any, batch: ty.BatchData) -> ty.BatchData:
        """Load ground-truth depth and store in loss targets. (y)"""
        batch[1]['depth'] = self._load_depth(data)
        return batch

    @abstractmethod
    def _load_depth(self, data: ty.Any) -> ty.A:
        """Load ground-truth depth from dataset."""

    def load_K(self, data: ty.Any, batch: ty.BatchData) -> ty.BatchData:
        """Load camera intrinsics and store in loss targets. (y)"""
        batch[1]['K'] = self._load_K(data)
        return batch

    @abstractmethod
    def _load_K(self, data: ty.Any) -> ty.A:
        """Load camera intrinsics from dataset."""

    def augment(self, batch: ty.BatchData) -> ty.BatchData:
        """Augment a dataset item. Currently supported are "horizontal flipping" and "colour jittering"."""
        if random.random() <= self.prob_flip: batch = self.apply_flip_aug(batch)
        if random.random() <= self.prob_photo: batch = self.apply_photo_aug(batch)
        return batch

    def apply_flip_aug(self, batch: ty.BatchData) -> ty.BatchData:
        """Apply horizontal flipping augmentation. All images are flipped, including "non-augmented" version in `y`."""
        x, y, m = batch
        self.logger.debug('Triggered Augmentation: Horizontal flip')
        m['augs'] += '[FlipLR]'

        x['imgs'], y['imgs'] = self.flip(x['imgs']), self.flip(y['imgs'])

        if self.supp_idxs:
            x['supp_imgs'], y['supp_imgs'] = self.flip(x['supp_imgs'], axis=-2), self.flip(y['supp_imgs'], axis=-2)
            if 'T_stereo' in y: y['T_stereo'][0, 3] *= -1

        if 'depth' in y: y['depth'] = self.flip(y['depth'])
        return batch

    def apply_photo_aug(self, batch: ty.BatchData) -> ty.BatchData:
        """Apply colour jittering augmentation to `x`. The same jittering is applied to target and support."""
        x, y, m = batch
        self.logger.debug('Triggered Augmentation: Photometric')
        m['augs'] += '[Photo]'

        imgs = x['imgs'][None]
        if self.supp_idxs: imgs = np.concatenate((imgs, x['supp_imgs']))

        imgs = ops.to_np(self.photo(ops.to_torch(imgs)))

        x['imgs'] = imgs[0]
        if self.supp_idxs: x['supp_imgs'] = imgs[1:]
        return batch

    def transform(self, batch: ty.BatchData) -> ty.BatchData:
        """Apply ImageNet standarization to `x`."""
        x = batch[0]
        x['imgs'] = ops.standardize(x['imgs'])
        if self.supp_idxs: x['supp_imgs'] = ops.standardize(x['supp_imgs'])
        return batch

    def create_axs(self) -> ty.Axes:
        """Create axes required for displaying."""
        _, axs = plt.subplots(1 + len(self.supp_idxs) + ('depth' in self.datum))
        if isinstance(axs, plt.Axes): axs = np.array([axs])
        plt.tight_layout()
        return axs

    def show(self, batch: ty.BatchData, axs: ty.Axes) -> None:
        """Show a single dataset item."""
        x, y, m = batch
        use_aug = True
        d = x if use_aug else y

        i = 0; axs[i].imshow(ops.unstandardize(d['imgs']) if use_aug else d['imgs'])
        if self.supp_idxs:
            for ax, im in zip(axs[1:], d['supp_imgs']):
                i += 1; ax.imshow(ops.unstandardize(im) if use_aug else im)

        if 'depth' in y: i += 1; axs[i].imshow(viz.rgb_from_disp(y['depth'], invert=True))

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import src.typing as ty
from src import get_logger, register
from src.datasets import BaseDataset
from src.external_libs import load_midas_net, load_newcrfs_net
from src.tools import geometry as geo, ops
from src.utils import io
from .trainer import MonoDepthModule

__all__ = ['MonoDepthPredictor', 'BenchmarkPredictor', 'NewcrfsPredictor', 'MidasPredictor']


# -----------------------------------------------------------------------------
class MonoDepthPredictor(ABC):
    """Base class for computing depth predictions over a datasets.

    A prediction consists of the following steps:
        1. Pre-processing: Resize, apply transforms...
        2. Forward: Compute models predictions and index output dict (or equivalent)
        3. Post-processing: Scale disparity, resize...

    The additional functions should be overwritten:
        :method get_img_shape: Return the image size based on the used dataset.
        :method load_model: Return a pretrained model based on the input args.
    """
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.logger = get_logger(cls.__qualname__)

    @abstractmethod
    def load_model(self, *args, **kwargs) -> nn.Module:
        """Load a pretrained model. Must be overriden."""

    @staticmethod
    def get_img_shape(data_type: str) -> ty.N[tuple[int, int]]:
        """Get image size depending on the dataset type. `None` loads the original size for each dataset."""
        return None

    def preprocess(self, imgs: ty.T) -> ty.T:
        """Prepare dataset images for network forward pass. Default no-op.

        In the base case, we likely don't need to do anything.
        This is more useful for external models/checkpoints that have different requirements (e.g. Miaas).

        :param imgs: (Tensor) (b, 3, h, w) Dataset images (already resized and standardized).
        :return: (Tensor) (b, 3, h, w) Preprocessed images.
        """
        return imgs

    def forward(self, net: nn.Module, imgs: ty.T) -> ty.T:
        """Run a network forward pass. Default is vanilla fwd.

        :param net: (nn.Module) Pretrained network to run.
        :param imgs: (Tensor) Input images.
        :return: (Tensor) (b, *1, h, w) Predicted raw depth/disparity.
        """
        return net(imgs)

    def postprocess(self, pred: ty.T, imgs: ty.T) -> ty.T:
        """Apply additional transformations to prediction. Default is no-op.

        :param pred: (Tensor) (b, *1, h, w) Raw network predictions.
        :param imgs: (Tensor) (b, 3, h, w) Dataset images prior to pre-processing (e.g. as reference shape).
        :return: (Tensor) (b, 1, h, w) Post-processed DISPARITY. Must be 4-D.
        """
        return pred

    def forward_batch(self, x: dict, net: nn.Module, device: torch.device, use_stereo_blend: bool = False) -> ty.T:
        imgs = self.preprocess(x['imgs'])
        pred = self.forward(net, imgs.to(device))

        if use_stereo_blend:
            imgs_flip = self.preprocess(x['imgs'].flip(dims=[-1]))
            pred_flip = self.forward(net, imgs_flip.to(device))
            pred = geo.blend_stereo(pred, pred_flip.flip(dims=[-1]))

        pred = self.postprocess(pred, x['imgs'])
        return pred

    def apply(self,
              net: nn.Module,
              dl: DataLoader,
              func: ty.Callable,
              use_stereo_blend: bool = False,
              device: ty.N[str] = None,
              *args, **kwargs) -> None:
        """Compute predictions for entire dataset and apply an additional function to each batch.

        :param net: (nn.Module) Pretrained network to make predictions.
        :param dl: (DataLoader) Dataset to compute prediction on. Should be subclass of `BaseDataset`.
        :param func: (Callable) Function to apply to each batch. Must accept `(batch, pred, *args, **kwargs)`.
        :param use_stereo_blend: (bool) If `True`, apply "flip & average" augmentations. Not recommended.
        :param device: (None|str) Device on which to compute predictions.
        :return: (ndarray) (b, h, w) Dataset predictions.
        """
        torch.backends.cudnn.benchmark = True
        if not isinstance(dl.dataset, BaseDataset): raise TypeError('DataLoader must use a `src.dataset.BaseDataset`!')

        device = ops.get_device(device)
        net = net.to(device)

        for i, batch in enumerate(tqdm(dl)):
            pred = self.forward_batch(batch[0], net, device, use_stereo_blend)
            assert pred.ndim == 4, f"Prediction must be 4-D, got {pred.shape}!"
            pred = pred.detach().cpu()
            func(batch, pred, *args, **kwargs)

    def __call__(self,
                 net: nn.Module,
                 dl: DataLoader,
                 use_stereo_blend: bool = False,
                 device: ty.N[str] = None) -> ty.A:
        """Compute network predictions for a whole dataset.

        :param net: (nn.Module) Pretrained network to make predictions.
        :param dl: (DataLoader) Dataset to compute prediction on. Should be subclass of `BaseDataset`.
        :param use_stereo_blend: (bool) If `True`, apply "flip & average" augmentations. Not recommended.
        :param device: (None|str) Device on which to compute predictions.
        :return: (ndarray) (b, h, w) Dataset predictions.
        """
        torch.backends.cudnn.benchmark = True
        if not isinstance(dl.dataset, BaseDataset): raise TypeError('DataLoader must use a `src.dataset.BaseDataset`!')

        device = ops.get_device(device)
        net = net.to(device)

        b = dl.batch_size

        preds = torch.zeros(len(dl.dataset), 1, dl.dataset.h, dl.dataset.w)
        for i, (x, *_) in enumerate(tqdm(dl)):
            pred = self.forward_batch(x, net, device, use_stereo_blend)
            assert pred.ndim == 4, f"Prediction must be 4-D, got {pred.shape}!"
            preds[i*b:i*b+b] = pred.detach().cpu()

        sanity = preds.flatten(1, -1).sum(1)
        if len(idxs := (sanity == 0).nonzero()): raise ValueError(f"Found empty predictions at indices '{idxs}'!")

        preds = preds.squeeze().numpy()
        return preds
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
@register('ours')
class BenchmarkPredictor(MonoDepthPredictor):
    """Class to run predictions for models from this benchmark."""

    @staticmethod
    def get_img_shape(data_type: str) -> tuple[int, int]:
        """Return fixed image size based on dataset. Keep width at 640 and resize height."""
        return {
            'ddad': (416, 640),
            'diode': (480, 640),
            'kitti': (192, 640),
            'kitti_lmdb': (192, 640),
            'mannequin': (384, 640),
            'mannequin_lmdb': (384, 640),
            'mapfree': (512, 384),
            'nyud': (480, 640),
            'sintel': (288, 640),
            'syns_patches': (192, 640),
            'tum': (480, 640),
        }[data_type]

    def load_model(self, ckpt_file: Path, cfg_files: ty.N[ty.S[Path]] = None) -> nn.Module:
        """Load pretrained model from this benchmark.

        :param ckpt_file: (Path) Path to pretrained checkpoint.
        :param cfg_files: (None|ty.S[Path]) Additional model cfg files (when loading incompatible legacy models).
        :return: (nn.Module) Pretrained model.
        """
        self.logger.info(f"Loading pretrained weights from '{ckpt_file}'...")
        mod = MonoDepthModule.load_from_checkpoint(ckpt_file, strict=False, cfg=io.load_merge_yaml(*cfg_files)) \
            if cfg_files else MonoDepthModule.load_from_checkpoint(ckpt_file, strict=True)

        self.min_depth, self.max_depth = getattr(mod, 'min_depth', None), getattr(mod, 'max_depth', None)

        mod.eval().freeze()
        net = next(v for k, v in mod.nets.items() if 'depth' in k)
        return net

    def forward(self, net: nn.Module, imgs: ty.T) -> ty.T:
        """Run a network forward pass. Return highest-resolution disparity prediction.

        :param net: (nn.Module) Pretrained network to run.
        :param imgs: (Tensor) Input images.
        :return: (Tensor) (b, 1, h, w) Predicted sigmoid disparity.
        """
        return net(imgs)['disp'][0]

    def postprocess(self, pred: ty.T, imgs: ty.T) -> ty.T:
        """Scale network sigmoid disparities.

        :param pred: (Tensor) (b, 1, h, w) Predicted sigmoid disparity.
        :param imgs: (Tensor) (b, 3, h, w) Dataset images prior to pre-processing.
        :return: (Tensor) (b, 1, h, w) Scaled disparity.
        """
        if self.min_depth or self.max_depth: pred = geo.to_scaled(pred, min=0.1, max=100)[0]
        return pred
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
@register('newcrfs')
class NewcrfsPredictor(MonoDepthPredictor):
    """Class to run NeWCRFs predictions. From https://arxiv.org/abs/2203.01502."""

    @staticmethod
    def get_img_shape(data_type: str) -> tuple[int, int]:
        """Return fixed image size based on dataset. Keep height at 352 (outdoor) or 480 (indoor) and resize width."""
        return {
            'ddad': (352, 576),
            'diode': (480, 640),
            'kitti': (352, 1216),
            'kitti_lmdb': (352, 1216),
            'mannequin': (384, 640),
            'mannequin_lmdb': (384, 640),
            'mapfree': (512, 384),
            'nyud': (480, 640),
            'sintel': (352, 800),
            'syns_patches': (352, 1216),
            'tum': (480, 640),
        }[data_type]

    def load_model(self, scene_type: str) -> nn.Module:
        """Load pretrained NeWCRFs model.

        :param scene_type: (str) Model variant to load. {indoor, outdoor}
        :return: (nn.Module) Pretrained model.
        """
        self.logger.info(f"Loading NeWCRFs weights for '{scene_type}'...")
        return load_newcrfs_net(scene_type).module

    def postprocess(self, pred: ty.T, imgs: ty.T) -> ty.T:
        """Convert NeWCRFS depth into disparity.

        :param pred: (Tensor) (b, 1, h, w) Predicted metric depth.
        :param imgs: (Tensor) (b, 3, h, w) Dataset images prior to pre-processing.
        :return: (Tensor) (b, 1, h, w) Metric disparity.
        """
        return geo.to_inv(pred)
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
@register('midas')
class MidasPredictor(MonoDepthPredictor):
    """Class to run Midas predictions. From https://arxiv.org/abs/1907.01341v3."""
    def load_model(self, name: str) -> nn.Module:
        """Load pretrained Midas/DPT model and transforms.

        :param name: (Path) Midas checkpoint to load. {MiDaS, DPT_Large, DPT_BEiT_L_512}
        :return: (nn.Module) Pretrained model.
        """
        self.logger.info(f"Loading MiDaS weights for '{name}'...")
        net, self.tfm = load_midas_net(name)
        return net

    def preprocess(self, imgs: ty.T) -> ty.T:
        """Apply Midas preprocessing transforms. Note that we need to undo standarization and convert into numpy.

        :param imgs: (Tensor) (b, 3, h, w) Dataset images.
        :return: (Tensor) (b, 3, h, w) Preprocessed images.
        """
        imgs = (ops.to_np(ops.unstandardize(imgs)) * 255).astype(np.uint8)
        imgs = torch.cat([self.tfm(i) for i in imgs])
        return imgs

    def postprocess(self, pred: ty.T, imgs: ty.T) -> ty.T:
        """Apply bicubic upsampling to match original resolution.

        :param pred: (Tensor) (b, h, w) Predicted scaleless disparity.
        :param imgs: (Tensor) (b, 3, h, w) Dataset images prior to pre-processing.
        :return: (Tensor) (b, 1, h, w) Scaleless disparity.
        """
        return ops.interpolate_like(pred[:, None], imgs, mode='bicubic', align_corners=False)
# -----------------------------------------------------------------------------

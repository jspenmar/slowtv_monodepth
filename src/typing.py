"""Collection of common type hints and aliases."""
# noinspection PyUnresolvedReferences
from numbers import Real
from pathlib import Path
# noinspection PyUnresolvedReferences
from typing import (
    Any, Callable, Generator, Iterable, Optional, Protocol, Sequence, Type, TypeVar, TypedDict, Union, final
)

import torch.nn as nn
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from torch import Tensor
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, Dataset

__all__ = [
    'U', 'N', 'S', 'T', 'A',
    'TimerData', 'Axes', 'BatchData', 'TensorDict', 'LossData', 'Shape',
    'Metrics', 'ArrDict', 'ModDict', 'DataDict', 'SchedDict', 'PredDict',
    'StrDict', 'PathDict', 'DictDict', 'FloatDict',
    'SuppImageNotFoundError',
    'DepthPred', 'PosePred', 'AutoencoderPred',
]

# Shorthand aliases.
U = Union
N = Optional
S = Sequence
T = Tensor
A = NDArray


# Used to indicate that the support images for the photometric loss are not available.
# Datasets based on MdeBaseDataset use this to retry and load a different item.
class SuppImageNotFoundError(FileNotFoundError): pass


class Predictor(Protocol):
    @staticmethod
    def get_img_shape(data_type: str) -> N[tuple[int, int]]: ...
    def __call__(self, net: nn.Module, dl: DataLoader, use_stereo_blend: bool, device: N[str]) -> NDArray: ...
    def apply(self, net: nn.Module, dl: DataLoader, func: Callable, use_stereo_blend: bool, device: N[str],
              *args, **kwargs) -> None: ...

    def load_model(self, *args, **kwargs) -> nn.Module: ...
    def preprocess(self, imgs: T) -> T: ...
    def forward(self, net: nn.Module, imgs: T) -> T: ...
    def postprocess(self, pred: Tensor, imgs: T) -> T: ...


Axes = Union[plt.Axes, A]
Shape = tuple[int, int]

Metrics = dict[str, U[str, float]]
ArrDict = dict[str, A]
StrDict = dict[str, str]
PathDict = dict[str, Path]
DictDict = dict[str, dict]
TensorDict = dict[U[str, int], T]
FloatDict = dict[str, float]
ModDict = dict[str, Type[nn.Module]]
DataDict = dict[str, Type[Dataset]]
SchedDict = dict[str, Type[_LRScheduler]]
PredDict = dict[str, Type[Predictor]]

TimerData = dict[str, U[int, float]]
BatchData = tuple[dict, dict, dict]
LossData = tuple[T, TensorDict]

# NETWORK OUTPUTS
class DepthPred(TypedDict, total=False):
    depth_feats: S[T]
    disp: dict[int, T]

    # Optional (see `DepthNet.forward`)
    disp_stereo: dict[int, T]
    mask: dict[int, T]
    mask_stereo: dict[int, T]


class PosePred(TypedDict, total=False):
    R: T
    t: T

    # Optional (see `PoseNet.forward`)
    fs: T
    cs: T


class AutoencoderPred(TypedDict, total=True):
    autoenc_feats: S[T]
    autoenc_imgs: dict[int, T]

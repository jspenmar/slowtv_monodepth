from collections import namedtuple
from pathlib import Path

import numpy as np

import src.typing as ty
from src.external_libs import ImageDatabase, LabelDatabase
from src.tools import T_from_Rt
from src.utils import io
from . import PATHS

# CONSTANTS
# -----------------------------------------------------------------------------
Item = namedtuple('MannequinItem', 'seq stem')
# -----------------------------------------------------------------------------


# PATH BUILDING
# -----------------------------------------------------------------------------
def get_split_file(mode: str) -> Path:
    """Get the split filename for the specified `mode`."""
    return PATHS['mannequin_lmdb']/'splits'/f'{mode}_files.txt'


def get_info_file(mode: str, seq: str) -> Path:
    """Get info filename with calibration and poses based on the mode and sequence."""
    return PATHS['mannequin_lmdb']/mode/seq/f'calibration.txt'


def get_imgs_path(mode: str) -> Path:
    """Get image LMDB filename based on the mode and sequence."""
    return PATHS['mannequin_lmdb']/mode/'images'


def get_depths_path(mode: str) -> Path:
    """Get image LMDB filename based on the mode and sequence."""
    return PATHS['mannequin_lmdb']/mode/'depths'


def get_shapes_path(mode: str) -> Path:
    """Get image LMDB filename based on the mode and sequence."""
    return PATHS['mannequin_lmdb']/mode/'shapes'


def get_intrinsics_path(mode: str) -> Path:
    """Get image LMDB filename based on the mode and sequence."""
    return PATHS['mannequin_lmdb']/mode/'intrinsics'


def get_poses_path(mode: str) -> Path:
    """Get image LMDB filename based on the mode and sequence."""
    return PATHS['mannequin_lmdb']/mode/'poses'
# -----------------------------------------------------------------------------


# LOADING
# -----------------------------------------------------------------------------
def load_split(mode: str) -> tuple[Path, ty.S[Item]]:
    """Load items (as [seq, stem]) in the specified split."""
    file = get_split_file(mode)
    items = io.tmap(Item, io.readlines(file, split=True), star=True)
    return file, items


def load_info(mode: str, seq: str) -> dict[str, dict[str, ty.A]]:
    """Load image shape, intrinsics and poses for each image in sequence based on the mode and sequence."""
    file = get_info_file(mode, seq)
    lines = io.readlines(file, split=True)

    n_imgs, offset = map(int, lines.pop(0))
    assert len(lines) == n_imgs*6

    items = {}
    for i in range(n_imgs):
        y_min, y_max, x_min, x_max = map(int, lines.pop(0))
        d = {'shape': (len(range(y_min, y_max)), len(range(x_min, x_max)))}

        fx, fy, cx, cy = map(float, lines.pop(0))
        d['K'] = np.array([
            [fx, 0, cx, 0],
            [0, fy, cy, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)

        R = np.array([io.tmap(float, lines.pop(0)) for _ in range(3)], dtype=np.float32)
        t = np.array(io.tmap(float, lines.pop(0)), dtype=np.float32)
        d['T'] = T_from_Rt(R, t)

        items[f'{i+offset:05d}'] = d

    assert not lines
    return items


def load_imgs(mode: str) -> ImageDatabase:
    """Load the image LMDB based on the mode and sequence."""
    path = get_imgs_path(mode)
    return ImageDatabase(path)


def load_depths(mode: str) -> LabelDatabase:
    """Load the image LMDB based on the mode and sequence."""
    path = get_depths_path(mode)
    return LabelDatabase(path)


def load_shapes(mode: str) -> LabelDatabase:
    """Load the image LMDB based on the mode and sequence."""
    path = get_shapes_path(mode)
    return LabelDatabase(path)


def load_intrinsics(mode: str) -> LabelDatabase:
    """Load the image LMDB based on the mode and sequence."""
    path = get_intrinsics_path(mode)
    return LabelDatabase(path)


def load_poses(mode: str) -> LabelDatabase:
    """Load the image LMDB based on the mode and sequence."""
    path = get_poses_path(mode)
    return LabelDatabase(path)
# -----------------------------------------------------------------------------

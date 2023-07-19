from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

import src.typing as ty
from src.tools import geometry as geo
from src.utils import io
from . import PATHS

# From official Sintel devkit.
# Check for endianness, based on Daniel Scharstein's optical flow code.
# Using little-endian architecture, these two should be equal.
TAG_FLOAT = 202021.25
TAG_CHAR = 'PIEH'


def create_splits() -> None:
    """Create train split based on all left camera files."""
    split_file = PATHS['sintel']/'splits'/'train_files.txt'
    io.mkdirs(split_file.parent)

    files = sorted((PATHS['sintel']/'train'/'camdata_left').glob('**/*.cam'))
    items = [f'{f.parent.stem} {f.stem}\n' for f in files]

    with open(split_file, 'w') as f: [f.write(i) for i in items]


@dataclass
class Item:
    """Class to load Sintel items. NOTE: We use the official TRAINING split as our TEST set."""
    mode: str  # {train}
    seq: str  # {seq}_{i}
    stem: str  # frame_{i:04}

    @classmethod
    def get_split_file(cls, mode: str) -> Path:
        """Get path to dataset split. {train}"""
        return PATHS['sintel']/'splits'/f'{mode}_files.txt'

    @classmethod
    def load_split(cls, mode: str) -> ty.S['Item']:
        """Load dataset split. {train}"""
        return [cls(mode, *s) for s in io.readlines(cls.get_split_file(mode), split=True)]

    def get_img_file(self) -> Path:
        """Get path to image file."""
        return PATHS['sintel']/self.mode/'final'/self.seq/f'{self.stem}.png'

    def get_depth_file(self) -> Path:
        """Get path to synthetic depth file."""
        return PATHS['sintel']/self.mode/'depth'/self.seq/f'{self.stem}.dpt'

    def get_cam_file(self) -> Path:
        """Get path to camera intrinsics/extrinsics file."""
        return PATHS['sintel']/self.mode/'camdata_left'/self.seq/f'{self.stem}.cam'

    def load_img(self) -> Image:
        """Load image."""
        return Image.open(self.get_img_file())

    def load_depth(self) -> ty.A:
        """Load synthetic depth map. Adapted from the official devkit."""
        with open(self.get_depth_file(), 'rb') as f:
            check = np.fromfile(f, dtype=np.float32, count=1)[0]
            assert check == TAG_FLOAT, f'Wrong tag in depth file ({check} vs. {TAG_FLOAT}). Big-endian machine?'
            w = np.fromfile(f, dtype=np.int32, count=1)[0]
            h = np.fromfile(f, dtype=np.int32, count=1)[0]
            numel = w*h
            assert w > 0 and h > 0 and numel > 1 and numel < 100000000, f'Wrong input size ({w=}, {h=})'
            depth = np.fromfile(f, dtype=np.float32, count=-1).reshape((h, w))[..., None]
        return depth

    def load_intrinsics(self) -> ty.A:
        """Load camera intrinsics. Adapted from the official devkit."""
        with open(self.get_cam_file(), 'rb') as f:
            check = np.fromfile(f, dtype=np.float32, count=1)[0]
            assert check == TAG_FLOAT, f'Wrong tag in cam file ({check} vs. {TAG_FLOAT}). Big-endian machine?'
            K = np.fromfile(f, dtype='float64', count=9).reshape((3, 3)).astype(np.float32)
            # T = np.fromfile(f, dtype='float64', count=12).reshape((3, 4)).astype(np.float32)  # Extrinsics

        K = geo.pad_K(K)
        return K

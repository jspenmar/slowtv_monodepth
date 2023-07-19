from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

import src.typing as ty
from src.utils import io
from . import PATHS


@dataclass
class Item:
    """Class to load items from DIODE dataset."""
    mode: str  # {val}
    split: str  # {indoors, outdoor}
    scene: str  # scene_{i:05}
    scan: str  # scan_{i:05}
    stem: str  # {scene}_{scan}_{split}_{i:03}_{i:03}_{type}

    @classmethod
    def get_split_file(cls, mode: str, split: str) -> Path:
        """Get path to split file based on mode {train, val} and scene type {indoors, outdoor}."""
        return PATHS['diode']/'data_list'/f'{mode}_{split}.csv'

    @classmethod
    def load_split(cls, mode: str, split: str) -> list['Item']:
        """Load split items based on mode {train, val} and scene type {indoors, outdoor}."""
        lines = io.readlines(cls.get_split_file(mode, split))
        lines = [Path(l.split(',')[0]) for l in lines]
        items = [Item(
            mode=parts[-5], split=parts[-4], scene=parts[-3], scan=parts[-2], stem=f.stem,
        ) for f in lines if (parts := f.parts)]  # "Hack" to get walrus in list comprehension
        return items

    # PATH BUILDING
    def get_img_file(self) -> Path:
        """Get path to item image file."""
        return PATHS['diode']/self.mode/self.split/self.scene/self.scan/f'{self.stem}.png'

    def get_depth_file(self) -> Path:
        """Get path to item LiDAR depth file."""
        return PATHS['diode']/self.mode/self.split/self.scene/self.scan/f'{self.stem}_depth.npy'

    def get_mask_file(self) -> Path:
        """Get path to item valid LiDAR mask file."""
        return PATHS['diode']/self.mode/self.split/self.scene/self.scan/f'{self.stem}_depth_mask.npy'

    # LOAD
    def load_img(self) -> Image:
        """Load image."""
        return Image.open(self.get_img_file())

    def load_depth(self) -> ty.A:
        """Load LiDAR depth."""
        return np.load(self.get_depth_file()).astype(np.float32)

    def load_mask(self) -> ty.A:
        """Load valid LiDAR mask."""
        return np.load(self.get_mask_file()).astype(bool)

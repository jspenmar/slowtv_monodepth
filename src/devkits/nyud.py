from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

import src.typing as ty
from src.utils import io
from . import PATHS


@dataclass
class Item:
    """Class to load items from the NYU Depth V2 dataset."""
    mode: str  # {test}
    stem: str  # {i:05}

    @classmethod
    def get_split_file(cls, mode: str) -> Path:
        """Get path to dataset split. {train, test}."""
        return PATHS['nyud']/'splits'/f'{mode}_files.txt'

    @classmethod
    def load_split(cls, mode: str) -> ty.S['Item']:
        """Load dataset split. {train, test}"""
        return [cls(mode, s) for s in io.readlines(cls.get_split_file(mode))]

    def get_img_file(self) -> Path:
        """Get path to image file."""
        return PATHS['nyud']/self.mode/'rgb'/f'{self.stem}.png'

    def get_depth_file(self) -> Path:
        """Get path to Kinect depth file."""
        return PATHS['nyud']/self.mode/'depth'/f'{self.stem}.npy'

    def load_img(self) -> Image:
        """Load image."""
        return Image.open(self.get_img_file())

    def load_depth(self) -> ty.A:
        """Load Kinect depth map."""
        return np.load(self.get_depth_file())

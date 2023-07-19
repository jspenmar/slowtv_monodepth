from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import src.typing as ty
from src.tools import geometry as geo
from src.utils import io
from . import PATHS


def create_split_file(mode: str = 'train') -> None:
    """Helper to create the files for each dataset split. {train, val, test}"""
    split_file = PATHS['mapfree']/'splits'/f'{mode}_files.txt'
    io.mkdirs(split_file.parent)

    files = sorted((PATHS['mapfree']/mode).glob('./*/seq?/*.jpg'))

    items = [f'{f.parent.parent.stem} {f.parent.stem} {f.stem}\n' for f in files]
    with open(split_file, 'w') as f: f.writelines(items)


@dataclass
class Item:
    """Class to load items from MapFreeReloc dataset."""
    mode: str  # {train, val, test}
    scene: str  # s{i:05}
    seq: str  # seq[0, 1]
    stem: str  # frame_{i:05}

    @classmethod
    def get_split_file(cls, mode: str) -> Path:
        """Get path to dataset split. {train, val, test}"""
        return PATHS['mapfree']/'splits'/f'{mode}_files.txt'

    @classmethod
    def load_split(cls, mode: str) -> ty.S['Item']:
        """Load dataset split. {train, val, test}"""
        return [cls(mode, *s) for s in io.readlines(cls.get_split_file(mode), split=True)]

    def get_img_file(self) -> Path:
        """Get path to image file."""
        return PATHS['mapfree']/self.mode/self.scene/self.seq/f'{self.stem}.jpg'

    def get_depth_file(self, src) -> Path:
        """Get path to depth file."""
        return PATHS['mapfree']/self.mode/self.scene/self.seq/f'{self.stem}.{src}.png'

    def get_intrinsics_file(self) -> Path:
        """Get path to intrinsics file. One per scene."""
        return PATHS['mapfree']/self.mode/self.scene/'intrinsics.txt'

    def get_poses_file(self) -> Path:
        """Get path to poses file. One per scene."""
        return PATHS['mapfree']/self.mode/self.scene/'poses.txt'

    def load_img(self) -> Image:
        """Load image."""
        return Image.open(self.get_img_file())

    def load_depth(self, src: str) -> ty.A:
        """Load depth, encoded in mm"""
        depth = cv2.imread(str(self.get_depth_file(src)), cv2.IMREAD_UNCHANGED)
        depth = depth[..., None].astype(np.float32) / 1000  # Encoded in mm
        return depth

    def load_intrinsics(self) -> ty.A:
        """Load intrinsics.
        Intrinsics are given as a single file per scene. We scan the file for the matching stem and load it.
        Not the most efficient, but it matches the interface of other datasets.
        """
        lines = io.readlines(self.get_intrinsics_file(), split=True)
        stem = f'{self.seq}/{self.stem}.jpg'

        line = next(l for l in lines if l[0] == stem)
        intrinsics = io.lmap(float, line[1:])  # As (fx, fy, cx, cy, w, h)

        K = np.zeros((4, 4), dtype=np.float32)
        (K[0, 0], K[1, 1], K[0, 2], K[1, 2]), K[2, 2], K[3, 3] = intrinsics[:-2], 1, 1
        return K

    def load_pose(self) -> ty.A:
        """Load poses.
        Poses are given as a single file per scene. They are represented as a quaternion (w, x, y, z) and translation
        (x, y, z). We scan the file for the matching stem and load it. Not the most efficient, but it matches the
        interface of other datasets.
        """
        lines = io.readlines(self.get_poses_file(), split=True)
        stem = f'{self.seq}/{self.stem}.jpg'

        line = next(l for l in lines if l[0] == stem)
        q, t = io.lmap(float, line[1:5]), io.lmap(float, line[5:8])  # As (w, x, y, z) and (x, y, z)

        T = geo.T_from_qt(np.array(q), np.array(t)).astype(np.float32)
        return T

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

import src.typing as ty
from src.utils import io
from . import PATHS


@dataclass
class Item:
    """Class to load items from TUM-RGBD dataset."""
    seq: str  # {rgbd_dataset_freiburg[2,3]_{seq}
    rgb_stem: str  # rgb/{timestamp}.png
    depth_stem: str  # depth/{timestamp}.png

    @classmethod
    def get_split_file(cls, mode: str) -> Path:
        """Get path to dataset split. {test}"""
        return PATHS['tum']/'splits'/f'{mode}_files.txt'

    @classmethod
    def load_split(cls, mode: str) -> ty.S['Item']:
        """Load dataset split. {test}"""
        file = cls.get_split_file(mode)
        return [cls(*line) for line in io.readlines(file, split=True)]

    def get_img_file(self) -> Path:
        """Get path to image file."""
        return PATHS['tum']/self.seq/self.rgb_stem

    def get_depth_file(self) -> Path:
        """Get path to Kinect depth file."""
        return PATHS['tum']/self.seq/self.depth_stem

    def load_img(self) -> Image:
        """Load image."""
        file = self.get_img_file()
        img = Image.open(file)
        return img

    def load_depth(self) -> ty.A:
        """Load Kinect depth map."""
        file = self.get_depth_file()
        depth = np.array(Image.open(file), dtype=np.float32) / 5000
        return depth[..., None]


def create_splits(th: float = 0.02, max: int = 2500, seed: int = 42) -> None:
    """Create a split of associated images & depth maps.

    :param th: (float) Maximum time difference between two images to be considered as associated.
    :param max: (int) Maximum number of images in split.
    :param seed: (int) Random seed.
    :return
    """
    file = PATHS['tum']/'splits'/'test_files.txt'
    io.mkdirs(file.parent)

    items = []
    seqs = io.get_dirs(PATHS['tum'], key=lambda f: f.stem != 'splits')
    for seq in seqs:
        img_file = seq/'rgb.txt'
        depths_file = seq/'depth.txt'

        first_list = read_file_list(img_file)
        second_list = read_file_list(depths_file)

        matches = associate(first_list, second_list, offset=0, max_difference=th)
        if 'freiburg2' in seq.stem: matches = matches[::3]
        for a, b in matches: items.append(f'{seq.stem} {first_list[a][0]} {second_list[b][0]}\n')

    random.seed(seed)
    random.shuffle(items)
    items = sorted(items[:max])

    with open(file, 'w') as f:
        f.writelines(items)


def read_file_list(filename):
    """Reads a trajectory from a text file. From: https://cvg.cit.tum.de/data/datasets/rgbd-dataset/tools

    File format:
    The file format is "stamp d1 d2 d3 ...", where stamp denotes the time stamp (to be matched)
    and "d1 d2 d3.." is arbitary data (e.g., a 3D position and 3D orientation) associated to this timestamp.

    Input:
    filename -- File name

    Output:
    dict -- dictionary of (stamp,data) tuples
    """
    with open(filename) as f:
        data = f.read()
        lines = data.replace(",", " ").replace("\t", " ").split("\n")
        list = [[v.strip() for v in line.split(" ") if v.strip() != ""] for line in lines if
                len(line) > 0 and line[0] != "#"]
        list = [(float(l[0]), l[1:]) for l in list if len(l) > 1]
    return dict(list)


def associate(first_list, second_list, offset, max_difference):
    """Associate image and depth pairs. From: https://cvg.cit.tum.de/data/datasets/rgbd-dataset/tools

    Associate two dictionaries of (stamp,data). As the time stamps never match exactly, we aim
    to find the closest match for every input tuple.

    Input:
    first_list -- first dictionary of (stamp,data) tuples
    second_list -- second dictionary of (stamp,data) tuples
    offset -- time offset between both dictionaries (e.g., to model the delay between the sensors)
    max_difference -- search radius for candidate generation

    Output:
    matches -- list of matched tuples ((stamp1,data1),(stamp2,data2))
    """
    first_keys = list(first_list.keys())
    second_keys = list(second_list.keys())
    potential_matches = [(abs(a - (b + offset)), a, b)
                         for a in first_keys
                         for b in second_keys
                         if abs(a - (b + offset)) < max_difference]
    potential_matches.sort()
    matches = []
    for diff, a, b in potential_matches:
        if a in first_keys and b in second_keys:
            first_keys.remove(a)
            second_keys.remove(b)
            matches.append((a, b))

    matches.sort()
    return matches

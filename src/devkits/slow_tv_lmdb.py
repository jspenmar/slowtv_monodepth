from collections import namedtuple
from pathlib import Path

import src.typing as ty
from src.external_libs import ImageDatabase, LabelDatabase
from src.utils import io
from . import PATHS

# CONSTANTS
# -----------------------------------------------------------------------------
Item = namedtuple('SlowTvItem', 'seq stem')
# -----------------------------------------------------------------------------


# PATH BUILDING
# -----------------------------------------------------------------------------
def get_split_file(mode: str, split: str) -> Path:
    """Get the split filename for the specified `mode`."""
    file = PATHS['slow_tv_lmdb']/'splits'/f'{split}'/f'{mode}_files.txt'
    return file


def get_category_file() -> Path:
    """Get filename containing list of video URLs."""
    return PATHS['slow_tv_lmdb']/'splits'/f'categories.txt'


def get_seqs() -> tuple[str]:
    """Get tuple of sequences names in dataset."""
    dirs = io.get_dirs(PATHS['slow_tv_lmdb'], key=lambda d: d.stem not in {'splits', 'videos', 'colmap'})
    dirs = io.tmap(lambda d: d.stem, dirs)
    return dirs


def get_imgs_path(seq: str) -> Path:
    """Get image LMDB filename based on the sequence."""
    return PATHS['slow_tv_lmdb']/seq


def get_calibs_path() -> Path:
    """Get calibration LMDB filename based on the sequence."""
    return PATHS['slow_tv_lmdb']/'calibs'
# -----------------------------------------------------------------------------


# LOADING
# -----------------------------------------------------------------------------
def load_categories(subcats: bool = True) -> list[str]:
    """Load list of categories per SlowTV scenes."""
    file = get_category_file()
    lines = [line.lower() for line in io.readlines(file)]
    if not subcats: lines = [line.split('-')[0] for line in lines]
    return lines


def load_split(mode: str, split: str) -> tuple[Path, ty.S[Item]]:
    """Load the split filename and items as (seq, stem)."""
    file = get_split_file(mode, split)
    items = io.tmap(Item, io.readlines(file, split=True), star=True)
    return file, items


def load_imgs(seq: str) -> ImageDatabase:
    """Load the image LMDB based on the mode and sequence."""
    path = get_imgs_path(seq)
    return ImageDatabase(path)


def load_calibs() -> LabelDatabase:
    """Load the image LMDB based on the mode and sequence."""
    path = get_calibs_path()
    return LabelDatabase(path)
# -----------------------------------------------------------------------------

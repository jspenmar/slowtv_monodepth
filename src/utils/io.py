"""Collection of reading, writing and conversion utilities."""
import itertools
from functools import partial
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

import src.typing as ty

__all__ = [
    'readlines', 'splitlines',
    'lmap', 'tmap', 'smap', 'mymap',
    'pil2np', 'np2pil',
    'write_yaml', 'load_yaml', 'load_merge_yaml',
]


# TEXT
# ------------------------------------------------------------------------------
def readlines(file: Path, /,
              encoding: str = None,
              split: bool = False,
              sep: ty.N[str] = None) -> ty.U[list[str], list[list[str]]]:
    """Read file as a list of strings."""
    with open(file, encoding=encoding) as f:
        lines = f.read().splitlines()
        if split: lines = splitlines(lines, sep)
        return lines


def splitlines(lines: list[str], sep: ty.N[str] = None) -> list[list[str]]:
    """Split each line in a list of lines."""
    return [l.split(sep) for l in lines]
# ------------------------------------------------------------------------------


# MAPPING
# ------------------------------------------------------------------------------
T = ty.TypeVar('T')


def mymap(fn: str, iterable: ty.Iterable, type: ty.N[T] = list, **kwargs) -> ty.U[ty.Generator, T]:
    """Apply instance method `fn` to each item in `iterable`.

    :param fn: (str) Function name to search as an attribute of each item.
    :param iterable: (Iterable) Iterable to apply function to.
    :param type: (None|type) If provided, convert output generator into this sequence type, e.g. list, tuple, set.
    :param kwargs: (dict) Additional kwargs to forward to `fn`.
    :return: (Iterable) Iterable mapped to the provided function.
    """
    if not isinstance(fn, str): raise TypeError(f"`fn` must be a str to search as an attribute of each item!")
    gen = (getattr(item, fn)(**kwargs) for item in iterable)
    return type(gen) if type else gen


def _map(fn: ty.Callable, *iterables: ty.Iterable, type: T, star: bool = False) -> T:
    """Map `fn` to each iterable item and convert to the specified container `type`."""
    map_fn = itertools.starmap if star else map
    return type(map_fn(fn, *iterables))


lmap = partial(_map, type=list)
tmap = partial(_map, type=tuple)
smap = partial(_map, type=set)
# ------------------------------------------------------------------------------


# FILES
# ------------------------------------------------------------------------------
Key = ty.Callable[[Path], bool]


def iterdir(path: Path, key: ty.N[Key] = None) -> list[Path]:
    """Get sorted contents in path, optionally filtered by the `key`."""
    key = key or (lambda f: True)
    return sorted(filter(key, path.iterdir()))


def get_dirs(path: Path, key: ty.N[Key] = None) -> list[Path]:
    """Get sorted directories in a path, optionally filtered by the `key`."""
    _key = lambda p: (p.is_dir() and (key(p) if key else True))
    return iterdir(path, _key)


def get_files(path: Path, key: ty.N[Key] = None) -> list[Path]:
    """Get sorted files in a path, optionally filtered by the `key`."""
    _key = lambda p: (p.is_file() and (key(p) if key else True))
    return iterdir(path, _key)


def has_contents(path: Path) -> bool:
    """Check if directory is not empty."""
    return path.is_dir() and bool(iterdir(path))


def mkdirs(*paths: Path, exist_ok: bool = True, parents: bool = True, **kwargs) -> None:
    """Create all input directories with laxer defaults."""
    for p in paths: p.mkdir(exist_ok=exist_ok, parents=parents, **kwargs)
# ------------------------------------------------------------------------------


# IMAGE CONVERSION
# ------------------------------------------------------------------------------
def pil2np(img: Image, /) -> ty.A:
    """Convert PIL image [0, 255] into numpy [0, 1]."""
    return np.array(img, dtype=np.float32) / 255.  # Default is float64!


def np2pil(arr: ty.A, /) -> Image:
    """Convert numpy image [0, 1] into PIL [0, 255]."""
    if arr.dtype == np.uint8: return Image.fromarray(arr)

    assert arr.max() <= 1
    return Image.fromarray((arr*255).astype(np.uint8))
# ------------------------------------------------------------------------------


# YAML
# ------------------------------------------------------------------------------
def write_yaml(file: Path, data: dict, mkdir: bool = False, sort_keys: bool = False) -> None:
    """Write data to a yaml file."""
    file = Path(file).with_suffix('.yaml')
    if mkdir: mkdirs(file.parent)
    with open(file, 'w') as f: yaml.dump(data, f, sort_keys=sort_keys)


def load_yaml(file: Path, loader: ty.N[yaml.Loader] = yaml.FullLoader) -> dict:
    """Load a single yaml file."""
    with open(file) as f: return yaml.load(f, Loader=loader)


def load_merge_yaml(*files: Path) -> dict:
    """Load a list of YAML configs and recursively merge into a single config.

    Following dictionary merging rules, the first file is the "base" config, which gets updated by the second file.
    We chain this rule for however many cfg we have, i.e. ((((1 <- 2) <- 3) <- 4) ... <- n)

    :param files: (Sequence[PathLike]) List of YAML config files to load, from "oldest" to "newest".
    :return:  (dict) The merged config from all given files.
    """
    old, *datas = [load_yaml(file) for file in files]
    for new in datas: old = _merge_yaml(old, new)  # Iteratively override with new cfg
    return old


def _merge_yaml(old: dict, new: dict) -> dict:
    """Recursively merge two YAML cfg.
    Dictionaries are recursively merged. All other types simply update the current value.

    NOTE: This means that a "list of dicts" will simply be updated to whatever the new value is,
    not appended to or recursively checked!

    :param old: (dict) Base dictionary containing default keys.
    :param new: (dict) New dictionary containing keys to overwrite in `old`.
    :return: (dict) The merge config.
    """
    d = old.copy()  # Just in case...
    for k, v in new.items():
        # If `v` is an existing dict, merge recursively. Otherwise replace/add `old`.
        d[k] = _merge_yaml(d[k], v) if k in d and isinstance(v, dict) else v
    return d
# ------------------------------------------------------------------------------

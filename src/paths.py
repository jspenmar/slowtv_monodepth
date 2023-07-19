"""Model & dataset path finder.

Objective is to provide a flexible way of managing paths to data and pretrained models.
By default, we assume data is stored in `/.../monodepth_benchmark/data`,
while models should be in `/.../monodepth_benchmark/models`.
Each user can provide a custom config file in `/path/to/repo/monodepth_benchmark/PATHS.yaml`
(which should not be tracked by Git...) with additional directories in which to find models/data.

Roots should be listed in order of preference. I.e. the first existing path will be given priority.
"""
import logging
import warnings
from pathlib import Path

import src.typing as ty
from src.utils import io

__all__ = ['MODEL_PATHS', 'DATA_PATHS', 'REPO_ROOT', 'MODEL_ROOTS', 'DATA_ROOTS', 'find_data_dir', 'find_model_file']


# HELPERS
# -----------------------------------------------------------------------------
Paths = list[Path]
_msg = "Additional roots file '{file}' does not exist! " \
       "To silence this warning, create the specified file with the following contents (without backquotes):\n" \
       "```\n" \
       "# -----------------------------------------------------------------------------\n" \
       "MODEL_ROOTS: []\n" \
       "DATA_ROOTS: []\n" \
       "# -----------------------------------------------------------------------------\n" \
       "```\n\n"


def _load_roots() -> tuple[Paths, Paths]:
    """Helper to load the additional model & data roots from the repo config."""
    file = REPO_ROOT/'PATHS.yaml'
    if not file.is_file():
        warnings.warn(_msg.format(file=file))
        return [], []

    paths = io.load_yaml(file)
    return io.lmap(Path, paths['MODEL_ROOTS']), io.lmap(Path, paths['DATA_ROOTS'])


def _build_paths(names: ty.StrDict, roots: Paths, key: str = '') -> ty.PathDict:
    """Helper to build the paths from a list of possible `roots`.
    NOTE: This returns the FIRST found path given by the order of roots. I.e. ordered by priority.
    """
    paths = {}
    for k, v in names.items():
        try:
            paths[k] = next(p for r in roots if (p := r/v).exists())
            logging.debug(f"Found {key} path '{k}': {paths[k]}")
        except StopIteration:
            logging.warning(f"No valid {key} path found for '{k}:{v}'!")

    return paths
# -----------------------------------------------------------------------------


# CONSTANTS
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parents[1]  # Path to `/.../monodepth_benchmark`

MODEL_ROOTS, DATA_ROOTS = _load_roots()
MODEL_ROOTS.append(REPO_ROOT/'models')
DATA_ROOTS.append(REPO_ROOT/'data')

models: ty.StrDict = {
    'newcrfs_indoor': 'newcrfs/model_nyu.ckpt',
    'newcrfs_outdoor': 'newcrfs/model_kittieigen.ckpt',
}


datas: ty.StrDict = {
    'ddad': 'DDAD',
    'diode': 'Diode',
    'kitti_depth': 'kitti_depth_benchmark',
    'kitti_raw': 'kitti_raw_sync',
    'kitti_raw_lmdb': 'kitti_raw_sync_lmdb',
    'mannequin': 'MannequinChallenge',
    'mapfree': 'mapfree',
    'mapfree_lmdb': 'mapfree_lmdb',
    'nyud': 'NYUD_v2',
    'sintel': 'Sintel',
    'slow_tv': 'slow_tv',
    'slow_tv_lmdb': 'slow_tv_lmdb',
    'syns_patches': 'syns_patches',
    'tum': 'TUM_RGBD',
}
# -----------------------------------------------------------------------------


# BUILD PATHS
# -----------------------------------------------------------------------------
MODEL_PATHS: ty.PathDict = _build_paths(models, MODEL_ROOTS, key='MODEL')
DATA_PATHS: ty.PathDict = _build_paths(datas, DATA_ROOTS, key='DATASET')


def find_model_file(name: str) -> Path:
    """Helper to find a model file in the available roots."""
    if (p := Path(name)).is_file(): return p

    try: return next(p for r in MODEL_ROOTS if (p := r/name).is_file())
    except StopIteration: raise FileNotFoundError(f"No valid path found for {name} in {MODEL_ROOTS}...")


def find_data_dir(name: str) -> Path:
    """Helper to find a dataset directory in the available roots."""
    if (p := Path(name)).is_dir(): return p

    try: return next(p for r in DATA_ROOTS if (p := r/name).is_file())
    except StopIteration: raise FileNotFoundError(f"No valid path found for {name} in {DATA_ROOTS}...")
# -----------------------------------------------------------------------------

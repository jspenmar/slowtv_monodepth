import sys
from pathlib import Path

# `dgp` path addition
sys.path.insert(0, str(Path(__file__).parents[1]/'external_libs'/'dgp'))

import src.typing as ty
from dgp.datasets import SynchronizedSceneDataset
from . import PATHS


def get_json_file() -> Path:
    """Path to the official DDAD config file."""
    return PATHS['ddad']/'ddad_train_val'/'ddad.json'


def get_dataset(mode: str, datum: ty.S[str]) -> SynchronizedSceneDataset:
    """Get the official DDAD dataset for the target split.

    :param mode: (str) Dataset split to load. {train, val}
    :param datum: (list[str]) DDAD data types to load. {camera_0[1-5], lidar}
    :return: (SynchronizedSceneDataset) DDAD dataset.
    """
    if mode not in {'train', 'val'}: raise ValueError(f"DDAD provides only train and val splits. Got '{mode}'.")

    return SynchronizedSceneDataset(
        str(get_json_file()),
        datum_names=datum,
        generate_depth_from_datum='lidar' if 'lidar' in datum else None,
        split=mode,
        use_diskcache=False,
    )

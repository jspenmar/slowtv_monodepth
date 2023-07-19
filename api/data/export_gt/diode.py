"""Script to export the DIODE ground-truth evaluation targets.

NOTE: We use the publicly available `val` split for evaluation, split into indoor and outdoor scenes.

This dataset will produce a targets file with the following variables.
    - depth: (b, h, w) Ground-truth depths.
"""
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from tqdm import tqdm

import src.typing as ty
from src.datasets import DiodeDataset


def save(file: Path, **kwargs) -> None:
    """Save a list of arrays as a npz file."""
    print(f"-> Saving to '{file}'...")
    np.savez_compressed(file,  **kwargs)


def export_diode(mode: str, scene: str, save_stem: ty.N[str] = None, overwrite: bool = False) -> None:
    """Export the ground truth LiDAR depth images for SYNS.

    :param mode: (str) Split mode to use. {'val'}
    :param scene: (str) Scene type to use. {'outdoor', 'indoor'}
    :param save_stem: (Optional[str]) Exported depth file stem (i.e. no suffix).
    :param overwrite: (bool) If `True`, overwrite existing exported files.
    """
    print(f"-> Exporting ground truth depths for DIODE '{mode}'...")
    ds = DiodeDataset(mode, scene, shape=None, as_torch=False)

    save_file = ds.split_file.parent/f'{save_stem}.npz'
    if not overwrite and save_file.is_file():
        raise FileExistsError(f"Target file '{save_file}' already exists. Set flag `--overwrite 1` to overwrite")

    depths = np.array([y['depth'].squeeze()*y['mask'] for _, y, _ in tqdm(ds)])
    save(save_file, depth=depths)


if __name__ == '__main__':
    parser = ArgumentParser('Script to export a target depth dataset as a npz file.')
    parser.add_argument('--mode', default='val', choices={'val'}, help='Split mode to use.')
    parser.add_argument('--scene', default='outdoor', choices={'outdoor', 'indoors'}, help='Split mode to use.')
    parser.add_argument('--save-stem', default=None, help='Exported targets file stem (i.e. without suffix).')
    parser.add_argument('--overwrite', default=0, type=int, help='If 1, overwrite existing exported files.')
    args = parser.parse_args()

    if args.save_stem is None: args.save_stem = f'targets_{args.mode}_{args.split}'
    export_diode(args.mode, args.scene, args.save_stem, args.overwrite)

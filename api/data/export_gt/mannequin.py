"""Script to export the Mannequin Challenge ground-truth evaluation targets.

This dataset will produce a targets file with the following variables.
    - depth: (b, h, w) Ground-truth depths.
    - K: (b, 4, 4) Camera intrinsic parameters.
"""
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from tqdm import tqdm

import src.typing as ty
from src.datasets import MannequinDataset
from src.tools import geometry as geo


def save(file: Path, **kwargs) -> None:
    """Save a list of arrays as a npz file."""
    print(f'-> Saving to "{file}"...')
    np.savez_compressed(file,  **kwargs)


def export_mannequin(mode: str, save_stem: ty.N[str] = None, overwrite: bool = False) -> None:
    """Export the ground truth LiDAR depth images for SYNS.

    :param mode: (str) Split mode to use.
    :param save_stem: (Optional[str]) Exported depth file stem (i.e. no suffix).
    :param overwrite: (bool) If `True`, overwrite existing exported files.
    """
    print(f"-> Exporting ground truth depths for Mannequin '{mode}'...")
    ds = MannequinDataset(mode, datum='image depth K', shape=None, as_torch=False)

    save_file = ds.split_file.parent/f'{save_stem}.npz'
    if not overwrite and save_file.is_file():
        raise FileExistsError(f"Target file '{save_file}' already exists. Set flag `--overwrite 1` to overwrite")

    depths, Ks = [], []
    for _, y, m in tqdm(ds):
        depths.append(y['depth'].squeeze())
        Ks.append(geo.resize_K(y['K'], y['depth'].shape[-2:], shape=MannequinDataset.SHAPE))

    save(save_file, depth=np.array(depths, dtype=object), K=np.array(Ks))


if __name__ == '__main__':
    parser = ArgumentParser('Script to export a target depth dataset as a npz file.')
    parser.add_argument('--mode', default='test', choices={'test'}, help='Split mode to use.')
    parser.add_argument('--save-stem', default=None, help='Exported targets file stem (i.e. without suffix).')
    parser.add_argument('--overwrite', default=0, type=int, help='If 1, overwrite existing exported files.')
    args = parser.parse_args()

    if args.save_stem is None: args.save_stem = f'targets_{args.mode}'
    export_mannequin(args.mode, args.save_stem, args.overwrite)

"""Script to export the Sintel ground-truth evaluation targets.

NOTE: We use the publicly available `train` split for evaluation.

This dataset will produce a targets file with the following variables.
    - depth: (b, h, w) Ground-truth depths.
    - K: (b, 4, 4) Camera intrinsic parameters.
"""
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.datasets.sintel import SintelDataset


def save(file: Path, **kwargs) -> None:
    """Save a list of arrays as a npz file."""
    print(f'\n -> Saving to "{file}"...')
    np.savez_compressed(file,  **kwargs)


def export_sintel(mode, save_stem: str = None, overwrite: bool = False) -> None:
    """Export the ground-truth synthetic depth images for Sintel.

    :param mode: (str) Split mode to use.
    :param save_stem: (str) Exported depth file stem (i.e. no suffix).
    :param overwrite: (bool) If `True`, overwrite existing exported files.
    """
    print(f"-> Exporting ground truth depths for Sintel '{mode}'...")
    ds = SintelDataset(mode=mode, as_torch=False)

    save_file = ds.split_file.parent / f'{save_stem}.npz'
    if not overwrite and save_file.is_file():
        raise FileExistsError(f"Target file '{save_file}' already exists. Set flag `--overwrite 1` to overwrite")

    depths, Ks = [], []
    for x, y, m in tqdm(ds):
        depths.append(y['depth'].squeeze())
        Ks.append(y['K'])

    save(save_file, depth=np.array(depths), K= np.array(Ks))


if __name__ == '__main__':
    parser = ArgumentParser('Script to export a target depth dataset as a npz file.')
    parser.add_argument('--mode', default='train', choices={'train'}, help='Mode to use.')
    parser.add_argument('--save-stem', default=None, help='Exported targets file stem (i.e. without suffix).')
    parser.add_argument('--overwrite', default=0, type=int, help='If 1, overwrite existing exported files.')
    args = parser.parse_args()

    if args.save_stem is None: args.save_stem = f'targets_{args.mode}'
    export_sintel(args.mode, args.save_stem, args.overwrite)

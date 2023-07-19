"""Script to export the NYUD ground-truth evaluation targets.

This dataset will produce a targets file with the following variables.
    - depth: (b, h, w) Ground-truth depths.
"""
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from tqdm import tqdm
from src.datasets.nyud import NyudDataset


def save(file: Path, **kwargs) -> None:
    """Save a list of arrays as a npz file."""
    print(f'\n -> Saving to "{file}"...')
    np.savez_compressed(file,  **kwargs)


def export_nyud(mode: str, save_stem: str, overwrite: bool = False) -> None:
    """Export the ground truth LiDAR depth images for NYUD.

    :param mode: (str) Split mode to use. {'test'}
    :param save_stem: (str) Exported depth file stem (i.e. no suffix).
    :param overwrite: (bool) If `True`, overwrite existing exported files.
    """
    ds = NyudDataset(mode=mode, as_torch=False)

    save_file = ds.split_file.parent / f'{save_stem}.npz'
    if not overwrite and save_file.is_file():
        raise FileExistsError(f'Target file "{save_file}" already exists. Set flag `--overwrite 1` to overwrite')

    depths = np.array([batch[1]['depth'].squeeze() for batch in tqdm(ds)])
    save(save_file, depth=depths)


if __name__ == '__main__':
    parser = ArgumentParser('Script to export a target depth dataset as a npz file.')
    parser.add_argument('--mode', default='test', choices={'test'}, help='Mode to use.')
    parser.add_argument('--save-stem', default=None, help='Exported targets file stem (i.e. without suffix).')
    parser.add_argument('--overwrite', default=0, type=int, help='If 1, overwrite existing exported files.')
    args = parser.parse_args()

    if args.save_stem is None: args.save_stem = f'targets_{args.mode}'
    export_nyud(args.mode, args.save_stem, args.overwrite)

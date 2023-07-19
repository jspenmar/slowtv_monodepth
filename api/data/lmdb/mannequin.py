"""Script to convert the MannequinChallenge dataset to LMDB.

LMDBs (http://www.lmdb.tech/doc/) should provide faster loading & less load on the filesytem.

NOTE: This process takes quite a while!
Results are cached (i.e. LMDBs aren't recomputed unless forced) so the script can be interrupted and restarted.
"""
import shutil
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from tqdm import tqdm

import src.devkits.mannequin as mc
from src import DATA_PATHS as PATHS, LOGGER
from src.external_libs import write_image_database, write_label_database
from src.utils import io


# DIRECTORY PARSING
# -----------------------------------------------------------------------------
def process_dataset(src_dir: Path, dst_dir: Path, overwrite: bool = False) -> None:
    """Process the entire MannequinChallenge dataset."""
    print(f"-> Copying splits directory '{dst_dir/'splits'}'...")
    shutil.copytree(src_dir/'splits', dst_dir/'splits', dirs_exist_ok=True)

    for mode in ('train', 'val', 'test'):
        process_mode(src_dir/mode, dst_dir/mode, overwrite)


def process_mode(src_dir: Path, dst_dir: Path, overwrite: bool = False) -> None:
    """Process a full MannequinChallenge mode, e.g. train or val."""
    calibs = {d.stem: mc.load_info(dst_dir.stem, d.stem) for d in tqdm(src_dir.iterdir())}

    export_intrinsics(src_dir, dst_dir/'intrinsics', calibs, overwrite)
    export_shapes(src_dir, dst_dir/'shapes', calibs, overwrite)
    export_poses(src_dir, dst_dir/'poses', calibs, overwrite)
    export_images(src_dir, dst_dir/'images', overwrite)
# -----------------------------------------------------------------------------


# LMDB CREATION
# -----------------------------------------------------------------------------
def export_intrinsics(src_dir: Path, dst_dir: Path, calibs: dict[str, dict], overwrite: bool = False) -> None:
    """Create camera intrinsics LMDB."""
    if not overwrite and dst_dir.is_dir():
        print(f"-> Intrinsics already exist for dir '{src_dir.stem}'")
        return

    all_Ks = {}
    for k, v in tqdm(calibs.items()):
        Ks = np.stack(vv['K'] for vv in v.values())
        are_equal = (Ks[0] == Ks).all(axis=(-2, -1))
        if not are_equal.all(): LOGGER.warning(f"Miss-matched Ks! {Ks[0]} {Ks[np.where(~are_equal)]}")

        all_Ks[k] = Ks[0]

    print(f"-> Exporting intrinsics for dir '{src_dir.stem}'")
    write_label_database(all_Ks, dst_dir/'intrinsics')


def export_shapes(src_dir: Path, dst_dir: Path, calibs: dict[str, dict], overwrite: bool = False) -> None:
    """Create image shapes LMDB."""
    if not overwrite and dst_dir.is_dir():
        print(f"-> Shapes already exist for dir '{src_dir.stem}'")
        return

    all_shapes = {}
    for k, v in tqdm(calibs.items()):
        shapes = np.stack(vv['shape'] for vv in v.values())
        if not (shapes[0] == shapes).all(): raise ValueError(f"Miss-matched shapes!")

        all_shapes[k] = shapes[0]

    print(f"-> Exporting shapes for dir '{src_dir.stem}'")
    write_label_database(all_shapes, dst_dir/'shapes')


def export_poses(src_dir: Path, dst_dir: Path, calibs: dict[str, dict], overwrite: bool = False) -> None:
    """Create camera poses LMDB."""
    if not overwrite and dst_dir.is_dir():
        print(f"-> Poses already exist for dir '{src_dir.stem}'")
        return

    print(f"-> Exporting poses for dir {src_dir.stem}")
    all_poses = {f'{k}/{kk}': vv['T'] for k, v in tqdm(calibs.items()) for kk, vv in v.items()}
    write_label_database(all_poses, dst_dir/'poses')


def export_images(src_dir: Path, dst_dir: Path, overwrite: bool = False) -> None:
    """Create images LMDB."""
    if not overwrite and dst_dir.is_dir():
        print(f"-> Images already exist for dir '{src_dir.stem}'")
        return

    print(f"-> Exporting images for dir '{src_dir.stem}'")
    files = {f'{d.stem}/{p.stem}': p for d in tqdm(io.get_dirs(src_dir))
             for p in io.get_files(d, key=lambda f: f.suffix == '.jpg')}
    write_image_database(files, dst_dir)
# -----------------------------------------------------------------------------


if __name__ == '__main__':
    parser = ArgumentParser(description='Script to convert the MannequinChallenge dataset to LMDB.')
    parser.add_argument('--overwrite', default=0, type=int, help='If 1, overwrite existing LMDBs.')
    args = parser.parse_args()

    process_dataset(PATHS['mannequin'], PATHS['mannequin_lmdb'], args.overwrite)

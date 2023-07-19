import shutil
from argparse import ArgumentParser
from multiprocessing import Pool
from pathlib import Path

from tqdm import tqdm

import src.devkits.slow_tv as stv
from src import DATA_PATHS as PATHS
from src.external_libs import write_image_database, write_label_database
from src.utils import io


def process_dataset(overwrite=False):
    src, dst = PATHS['slow_tv'], PATHS['slow_tv_lmdb']

    # Copy splits dir
    print(f"-> Copying splits directory '{dst/'splits'}'...")
    shutil.copytree(src/'splits', dst/'splits', dirs_exist_ok=True)

    # Export intrinsics LMDB
    export_intrinsics(dst, overwrite)

    # Export all sequences.
    args = [(src/seq, dst, overwrite) for seq in stv.get_seqs()]
    with Pool() as p: list(p.starmap(export_seq, tqdm(args)))
    # [export_lmdb(*arg) for arg in args]


def export_seq(path: Path, save_root: Path, overwrite: bool = False) -> None:
    """Convert SlowTV video into an LMDB."""
    seq = path.stem
    out_dir = save_root/seq
    if not overwrite and out_dir.is_dir():
        print(f'-> Skipping directory "{out_dir}"...')
        return

    print(f'-> Export LMDB for dir "{seq}"')
    paths = {p.stem: p for p in io.get_files(path, key=lambda f: f.suffix == '.png')}
    write_image_database(paths, out_dir)


def export_intrinsics(save_root: Path, overwrite: bool = False) -> None:
    """Export SlowTV intrinsics as an LMDB."""
    out_dir = save_root/'calibs'
    if not overwrite and out_dir.is_dir():
        print(f'-> Skipping LMDB calibrations...')
        return

    print(f'-> Exporting intrinsics "{save_root/"calibs"}"...')
    data = {seq: stv.load_intrinsics(seq) for seq in stv.get_seqs()}
    write_label_database(data, save_root/'calibs')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--overwrite', default=0, type=int, help='If 1, overwrite existing LMDBs.')
    args = parser.parse_args()

    process_dataset(args.overwrite)

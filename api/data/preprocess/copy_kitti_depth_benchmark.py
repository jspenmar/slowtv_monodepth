"""Script to copy the Kitti Benchmark depth maps into the Kitti Raw Sync folder structure."""
import shutil
from argparse import ArgumentParser
from pathlib import Path

from tqdm import tqdm

import src.devkits.kitti_raw as kr
from src import DATA_PATHS


def main(src, dst):
    TARGET_DIR = 'depth_benchmark'
    K_DEPTH, K_RAW = src, dst
    print(f'-> Exporting Kitti Benchmark from "{K_DEPTH}" to "{K_RAW}"...')

    ROOT = K_RAW/TARGET_DIR
    ROOT.mkdir(exist_ok=True)
    for seq in kr.SEQS: (ROOT/seq).mkdir(exist_ok=True)

    for mode in ('train', 'val'):
        for path in tqdm(sorted((K_DEPTH/mode).iterdir())):
            seq = next(s for s in kr.SEQS if path.stem.startswith(s))
            shutil.copytree(path, ROOT/seq/path.stem, dirs_exist_ok=True)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('src', type=Path, default=DATA_PATHS.get('kitti_depth'))
    parser.add_argument('dst', type=Path, default=DATA_PATHS['kitti_raw'])
    args = parser.parse_args()

    main(args.src, args.dst)

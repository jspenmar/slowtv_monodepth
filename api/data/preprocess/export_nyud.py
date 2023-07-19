from argparse import ArgumentParser
from pathlib import Path

import h5py
import numpy as np
import scipy.io as sio
from tqdm import tqdm

from src import DATA_PATHS as PATHS
from src.utils import io


def loadmat(file):
    """Conflict with specific matfile versions?"""
    f = h5py.File(file)
    arr = {k: np.array(v) for k, v in f.items()}
    return arr


def export_split(mode, idxs, data, dst):
    img_dir = dst/mode/'rgb'
    depth_dir = dst/mode/'depth'
    split_file = dst/'splits'/f'{mode}_files.txt'
    io.mkdirs(img_dir, depth_dir, split_file.parent)

    with open(split_file, 'w') as f:
        for i in tqdm(idxs):
            i -= 1
            stem = f'{i:05}'
            img = data['images'][i-1].transpose((2, 1, 0)).astype(np.float32) / 255.
            depth = data['depths'][i-1].T[..., None]

            io.np2pil(img).save(img_dir/f'{stem}.png')
            np.save(depth_dir/f'{stem}.npy', depth)
            f.write(stem+'\n')


def main(dst):
    data_file = dst/'nyu_depth_v2_labeled.mat'
    split_file = dst/'splits.mat'

    data = loadmat(data_file)
    splits = sio.loadmat(split_file)

    export_split('train', splits['trainNdxs'].squeeze(), data, dst)
    export_split('test', splits['testNdxs'].squeeze(), data, dst)

    data_file.unlink()
    split_file.unlink()


if __name__ == '__main__':
    parser = ArgumentParser(description='Export NYUD mats as individual files.')
    parser.add_argument('dst', type=Path, default=PATHS['nyud'])
    args = parser.parse_args()

    main(args.dst)

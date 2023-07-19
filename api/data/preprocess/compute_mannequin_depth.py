import shutil
import subprocess
from argparse import ArgumentParser
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src import DATA_PATHS as PATHS
from src.utils import io


def read_array(path):
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1, usecols=(0, 1, 2), dtype=int)
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze().astype(np.float32)


def export_split(split, src, dst, overwrite=False):
    print(f'-> Exporting "{split}" split...')
    dst = dst/split
    io.mkdirs(dst)

    seqs = io.get_dirs(src/split)
    # fails = {}
    # for s in tqdm(seqs):
    #     try:
    #         export_seq((s, dst/s.stem, overwrite))
    #     except Exception as e:
    #         fails[s.stem] = e

    dsts = [dst/s.stem for s in seqs]
    ovs = [overwrite for _ in seqs]

    with Pool(8) as p:
        for _ in tqdm(p.imap_unordered(export_seq, zip(seqs, dsts, ovs)), total=len(seqs)): pass
    return {}


def export_seq(args):
    try:
        src, dst, overwrite = args
        depth_dir = dst/'depths'
        if not overwrite and depth_dir.is_dir():
            print(f'-> Skipping "{src.parent.stem}" sequence "{src.stem}"...')
            return

        print(f'-> Exporting "{src.parent.stem}" sequence "{src.stem}"...')

        shutil.rmtree(dst, ignore_errors=True)
        io.mkdirs(dst)

        db_path = dst/'database.db'
        img_dir = dst/'images'
        sparse_dir = dst/'sparse'
        refined_dir = dst/'refined'
        dense_dir = dst/'dense'
        io.mkdirs(img_dir, sparse_dir, refined_dir, dense_dir)

        [shutil.copy(f, img_dir) for f in io.get_files(src, key=lambda f: f.suffix == '.jpg')]

        subprocess.call([
            'colmap', 'feature_extractor',
            '--ImageReader.single_camera', '1',
            '--ImageReader.default_focal_length_factor', '0.85',
            '--SiftExtraction.peak_threshold', '0.02',
            '--database_path', db_path,
            '--image_path', img_dir,
        ])

        subprocess.call([
            'colmap', 'exhaustive_matcher',
            '--SiftMatching.max_error', '3',
            '--SiftMatching.min_inlier_ratio', '0.3',
            '--SiftMatching.min_num_inliers', '30',
            '--SiftMatching.guided_matching', '1',
            '--database_path', db_path,
        ])

        subprocess.call([
            'colmap', 'mapper',
            '--Mapper.tri_merge_max_reproj_error', '3',
            '--Mapper.ignore_watermarks', '1',
            '--Mapper.filter_max_reproj_error', '2',
            '--database_path', db_path,
            '--image_path', img_dir,
            '--output_path', sparse_dir,
        ])

        subprocess.call([
            'colmap', 'bundle_adjuster',
            '--input_path', sparse_dir/'0',
            '--output_path', refined_dir,
        ])

        subprocess.call([
            'colmap', 'image_undistorter',
            '--input_path', refined_dir,
            '--image_path', img_dir,
            '--output_path', dense_dir,
            '--output_type', 'COLMAP',
            '--max_image_size', '1600',
        ])

        subprocess.call([
            'colmap', 'patch_match_stereo',
            '--PatchMatchStereo.window_radius', '5',
            '--PatchMatchStereo.num_samples', '15',
            '--PatchMatchStereo.geom_consistency_regularizer', '1',
            '--PatchMatchStereo.geom_consistency_max_cost', '1.5',
            '--PatchMatchStereo.filter_min_ncc', '0.2',
            '--PatchMatchStereo.filter_min_num_consistent', '3',
            '--PatchMatchStereo.geom_consistency', 'true',
            '--workspace_path', dense_dir,
            '--workspace_format', 'COLMAP',
        ])

        files = io.get_files(dense_dir/'stereo'/'depth_maps', key=lambda f: 'geometric' in str(f))
        [np.save(src/f'{f.name.split(".")[0]}.npy', read_array(f)) for f in files]
        io.mkdirs(depth_dir)
    except:
        pass


def main(root):
    dst = root / 'colmap'
    io.mkdirs(dst)

    splits = ['test']
    fails = {}
    for s in tqdm(splits):
        fails[s] = export_split(s, root, dst, overwrite=False)

    print(fails)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('root', type=Path, default=PATHS['mannequin'])
    args = parser.parse_args()

    main(args.src)

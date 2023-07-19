from argparse import ArgumentParser
from multiprocessing import Pool

from tqdm import tqdm

import src.devkits.slow_tv as stv
from src.paths import DATA_PATHS as PATHS
from src.utils import io


overwrite = False
write_settings = True

vid_dir = PATHS['slow_tv']/'videos'
colmap_dir = PATHS['slow_tv']/'colmap'

fps = '10'
trim = 300  # Trim first/last 5 mins
data_scale = 4  # N times more data than sampling at 1 FPS
n_keep = 100
per_interval = (n_keep * eval(fps)) // data_scale

p_train = 0.9
val_skip = 100

n_colmap_imgs = 200
colmap_interval = 1


def save_settings(**kwargs):
    io.write_yaml(PATHS['slow_tv']/'splits'/'config.yaml', kwargs)


def export_scene(args):
    vid_file, cat = args
    seq = vid_file.stem
    seq_dir = PATHS['slow_tv']/seq

    # Export frames and decimate
    stv.extract_frames(
        vid_file, save_dir=seq_dir,
        fps=fps, trim_start=trim, n_keep=n_keep, per_interval=per_interval,
        overwrite=overwrite
    )

    # Estimate COLMAP intrinsics
    seeds = [42, 195, 335, 558, 724]
    for seed in seeds:
        try:
            stv.estimate_intrinsics(
                seq_dir, save_root=colmap_dir,
                n_imgs=n_colmap_imgs, interval=colmap_interval,
                seed=seed, overwrite=overwrite
            )
            break
        except RuntimeError: print(f'-> Failed COLMAP intrinsics with seed "{seed}"...')
    else: raise RuntimeError(f'-> Tried {seeds} and they all failed!!')

    # Add to category, sequence and all split
    stv.add_frames_to_split(seq_dir, cat, seq, 'all', p_train=p_train, skip=val_skip)


def main(args):
    if write_settings:
        save_settings(
            fps=fps, trim=trim, data_scale=data_scale, n_keep=n_keep, per_interval=per_interval,
            p_train=p_train, val_skip=val_skip,
            n_colmap_imgs=n_colmap_imgs, colmap_interval=colmap_interval,
        )

    cats = stv.load_categories(subcats=False)
    video_files = io.get_files(vid_dir)
    assert len(cats) == len(video_files), 'Non-matching SlowTV videos and labelled categories.'

    if args.idx is not None:
        export_scene((video_files[args.idx], cats[args.idx]))
        return

    if args.n_proc == 0:
        [export_scene(args) for args in zip(video_files, cats)]
    else:
        with Pool(args.n_proc) as p:
            list(tqdm(p.imap_unordered(export_scene, zip(video_files, cats)), total=len(cats)))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--idx', type=int, help='Index of video to export. If not specified, export all.')
    parser.add_argument('--n-proc', default=0, type=int, help='Number of multiprocessing processes. 0 to disable.')
    main(parser.parse_args())

import random
import shutil
import subprocess
from collections import namedtuple
from pathlib import Path

import numpy as np
import pytorch_lightning
from tqdm import tqdm

import src.typing as ty
from src.utils import io
from . import PATHS

# CONSTANTS
# -----------------------------------------------------------------------------
Item = namedtuple('SlowTvItem', 'seq stem')
# -----------------------------------------------------------------------------


# PATH BUILDING
# -----------------------------------------------------------------------------
def get_url_file() -> Path:
    """Get filename containing list of video URLs."""
    return PATHS['slow_tv']/'splits'/f'urls.txt'


def get_category_file() -> Path:
    """Get filename containing list of video URLs."""
    return PATHS['slow_tv']/'splits'/f'categories.txt'


def get_vid_files() -> list[Path]:
    """Get list of video filenames."""
    return sorted(f for f in (PATHS['slow_tv']/'videos').iterdir() if f.suffix == '.mp4')


def get_split_file(mode: str, split: str) -> Path:
    """Get the split filename for the specified `mode`."""
    file = PATHS['slow_tv']/'splits'/f'{split}'/f'{mode}_files.txt'
    return file


def get_seqs() -> tuple[str]:
    """Get tuple of sequences names in dataset."""
    dirs = io.get_dirs(PATHS['slow_tv'], key=lambda d: d.stem not in {'splits', 'videos', 'colmap'})
    dirs = io.tmap(lambda d: d.stem, dirs)
    return dirs


def get_intrinsics_file(seq: str) -> Path:
    """Get intrinsics filename based on the sequence."""
    return PATHS['slow_tv']/seq/f'intrinsics.txt'


def get_img_file(seq:str, stem: ty.U[str, int]) -> Path:
    """Get image filename based on the sequence and item number."""
    return PATHS['slow_tv']/seq/f'{int(stem):010}.png'
# -----------------------------------------------------------------------------


# LOADING
# -----------------------------------------------------------------------------
def load_categories(subcats: bool = True) -> list[str]:
    """Load list of categories per SlowTV scenes."""
    file = get_category_file()
    lines = [line.lower() for line in io.readlines(file)]
    if not subcats: lines = [line.split('-')[0] for line in lines]
    return lines


def load_split(mode: str, split: str) -> tuple[Path, ty.S[Item]]:
    """Load the split filename and items as (seq, stem)."""
    file = get_split_file(mode, split)
    items = io.tmap(Item, io.readlines(file, split=True), star=True)
    return file, items


def load_intrinsics(seq: str) -> ty.A:
    """Load intrinsics as 4x4 matrix based on the sequence."""
    file = get_intrinsics_file(seq)
    return np.loadtxt(file, dtype=np.float32)
# -----------------------------------------------------------------------------


# DATASET CREATION
# -----------------------------------------------------------------------------
def _non_uniform_decimate(seq: ty.S, n_keep: int, per_interval: int) -> list:
    """Keep `n_keep` items out of every `per_interval` items in the input sequence."""
    sentinel = None

    pad = per_interval - len(seq) % per_interval
    arr = np.append(seq, [sentinel] * pad)

    out = arr.reshape(-1, per_interval)[:, :n_keep].reshape(-1)
    out = out[out != sentinel]
    return out.tolist()


def extract_frames(vid_file: Path,
                   save_dir: ty.N[Path] = None,
                   fps: str = '1/1',
                   trim_start: int=300,
                   trim_end: ty.N[int] = None,
                   n_keep: ty.N[int] = None,
                   per_interval: ty.N[int] = None,
                   overwrite: bool = False) -> None:
    """Convert video into individual frames.

    :param vid_file: (Path) Video file to convert.
    :param save_dir: (None|Path) Export directory. (Default is same as video file)
    :param fps: (str) Frames per second to extract (ffmpeg format). (Default is 1 FPS)
    :param trim_start: (int) Frames to remove from start of the video.
    :param trim_end: (None) Frames to remove from end of the video. (Default is same as `trim_start`)
    :param n_keep: (None|int) Number of frames to keep. (Default is all)
    :param per_interval: (None|int) Interval over which to keep `n_keep` frames. (Default is all)
    :param overwrite: (bool) If `True`, overwrite existing frames.
    :return:
    """
    duration = float(subprocess.check_output([
        'ffprobe', '-v', '0', '-show_entries', 'format=duration', '-of', 'compact=p=0:nk=1', vid_file
    ]))
    trim_end = duration - (trim_end or trim_start)

    save_dir = save_dir or vid_file.parent/vid_file.stem
    if not overwrite and io.has_contents(save_dir):
        print(f'-> Skipping video "{vid_file}"...')
        return
    shutil.rmtree(save_dir, ignore_errors=True)
    io.mkdirs(save_dir)

    print(f'-> Exporting video to "{save_dir}" with trim "{trim_start}" "{trim_end}"...')
    subprocess.call([
        'ffmpeg', '-i', vid_file, '-r', fps, '-vf', f'trim={trim_start}:{trim_end}', save_dir/'%010d.png'
    ])

    if n_keep and per_interval: decimate_frames(save_dir, n_keep, per_interval)


def decimate_frames(path, n_keep, per_interval):
    """Decimate SlowTV video frames."""
    assert n_keep < per_interval
    print(f'-> Decimating "{path}" to "{n_keep}" per "{per_interval}" frames...')
    fs = io.get_files(path)
    fs_keep = _non_uniform_decimate(fs, n_keep, per_interval)
    fs_del = set(fs) - set(fs_keep)
    [f.unlink() for f in tqdm(fs_del)]


def add_frames_to_split(path: Path, *splits: str, p_train: float = 0.9, skip: int = 100) -> None:
    """Add SlowTV video frames to the desired split(s)."""
    seq = path.stem

    fs = [f.stem for f in io.get_files(path, key=lambda f: f.suffix == '.png')]
    n = int(len(fs) * p_train)
    train_fs, val_fs = fs[:n-skip], fs[n+skip:]

    for split in splits:
        train_file = get_split_file('train', split.lower())
        val_file = get_split_file('val', split.lower())
        io.mkdirs(train_file.parent, val_file.parent)

        print(f'-> Adding seq "{seq}" to split "{split}": {n-skip} - {len(fs)-n-skip}')
        with open(train_file, 'a') as ft: [ft.write(f'{seq} {i}\n') for i in train_fs]
        with open(val_file, 'a') as fv: [fv.write(f'{seq} {i}\n') for i in val_fs]


def estimate_intrinsics(path: Path,
                        save_root: Path,
                        n_imgs: int = 200,
                        interval: int = 1,
                        seed: int = 42,
                        overwrite: bool = False) -> None:
    """Estimate sequence intrinsics using Colmap and a small subset of frames.

    :param path: (Path) Path to directory containing frames.
    :param save_root: (Path) Root directory to save Colmap reconstructions.
    :param n_imgs: (int) Number of images to use for intrinsics estimation.
    :param interval: (int) Skip between frames.
    :param seed: (int) Random seed.
    :param overwrite: (bool) If `True`, overwrite existing reconstructions.
    :return:
    """
    pytorch_lightning.seed_everything(seed)
    seq = path.stem
    files = io.iterdir(path)

    skip = n_imgs * interval
    start = random.randint(0, len(files)-skip)
    files = files[start:start + skip:interval]

    out_dir = save_root/seq
    if not overwrite and out_dir.is_dir():
        print(f'-> Skipping directory "{out_dir}"...')
        return

    shutil.rmtree(out_dir, ignore_errors=True)

    try:
        print(f'-> Estimating Colmap intrinsics for seq "{seq}"')
        db_path = out_dir / 'database.db'
        img_dir = out_dir / 'images'
        sparse_dir = out_dir / 'sparse'
        txt_dir = out_dir / 'txt'
        io.mkdirs(img_dir, sparse_dir, txt_dir)

        [shutil.copy(f, img_dir) for f in files]

        subprocess.run([
            'colmap', 'feature_extractor', '--ImageReader.single_camera', '1',  # Same cam for whole sequence.
            '--database_path', db_path, '--image_path', img_dir,
        ])

        subprocess.run([
            'colmap', 'sequential_matcher', '--database_path', db_path
        ])

        subprocess.run([
            'colmap', 'mapper', '--database_path', db_path, '--image_path', img_dir, '--output_path', sparse_dir
        ])

        subprocess.run([
            'colmap', 'model_converter', '--input_path', sparse_dir / '0', '--output_path', txt_dir,
            '--output_type', 'TXT'
        ])

        cam = [line for line in io.readlines(txt_dir / 'cameras.txt') if not line.startswith('#')]
        assert len(cam) == 1

        cam = io.lmap(float, cam[0].split()[2:])  # As [CAM_IDX, CAM_MODEL, *CAM_PARAMS]
        assert cam[:2] == [1280, 720]

        w, h, f, cx, cy, r = cam
        np.savetxt(get_intrinsics_file(seq), np.array([
            [f, r, cx, 0],
            [r, f, cy, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]))

    except Exception as e:
        shutil.rmtree(out_dir, ignore_errors=True)
        raise RuntimeError(f'Failed on seq "{seq}": {e}')
# -----------------------------------------------------------------------------

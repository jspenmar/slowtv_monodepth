from pathlib import Path

import cv2
import numpy as np
import skimage.transform as skit
from PIL import Image
from matplotlib import pyplot as plt

import src.devkits.kitti_raw as kr
import src.typing as ty
from src import register
from src.tools import geometry as geo, viz
from . import MdeBaseDataset

__all__ = ['KittiRawDataset']


@register('kitti')
class KittiRawDataset(MdeBaseDataset):
    """Kitti Raw dataset.

    Datum:
        - Image: Target image from which to predict depth.
        - Support: Adjacent frames (either monocular or stereo) used to compute photometric consistency losses.
        - Depth: Target ground-truth benchmark depth (corrected LiDAR).
        - Depth Velo: Target ground-truth velodyne depth (raw LiDAR).
        - Depth Hint: Hand-crafted fused SGBM depth estimates.
        - K: Camera intrinsic parameters.

    See BaseDataset for additional added metadata.

    Batch:
        x: {
            imgs: (Tensor) (3, h, w) Augmented target image.
            supp_imgs: (Tensor) (n, 3, h, w) Augmented support frames.
            supp_idxs: (Tensor) (n,) Indexes of the support frames relative to target.
        }

        y: {
            imgs: (Tensor) (3, h, w) Non-augmented target image.
            supp_imgs: (Tensor) (n, 3, h, w) Non-augmented support frames.
            depth: (Tensor) (1, h, w) Ground-truth target depth  (either Benchmark or LiDAR)
            depth_hints: (Tensor) (1, h, w) Fused SGBM depth hints.
            T_stereo: (Tensor) (4, 4) Transform to the stereo support pair.
            K: (Tensor) (4, 4) Camera intrinsics.
        }

        m: {
            stem: (str) Path to loaded item.
            supp: (str) Support frame multiplier.
        }

    Parameters:
    :param split: (str) Kitti split to use. {eigen, eigen_zhou, eigen_benchmark, odometry...}
    :param mode: (str) Training mode to use. {train, val, test}

    Attributes:
    :attr K: (ty.A) (4, 4) Averaged camera intrinsics, normalized based on `shape`.
    :attr split_file: (Path) File containing the list of items in the loaded split.
    :attr items_data: (list[kr.Item]) List of dataset items as (seq, cam, stem).
    """
    VALID_DATUM = 'image support depth depth_velo depth_hint K'
    SHAPE = 376, 1242

    def __init__(self, split: str, mode: str, **kwargs):
        super().__init__(**kwargs)
        self.split = split
        self.mode = mode

        # NOTE: This might seem counterintuitive, but it makes sense.
        # This transform represents the direction in which the PIXELS move in, NOT the camera.
        self._cam2sign = {'image_02': -1, 'image_03': 1}
        self._side2cam = {'l': 'image_02', 'r': 'image_03'}
        self._cam2stereo = {'image_02': 'image_03', 'image_03': 'image_02'}

        self.K = geo.resize_K(np.array([
            [0.58, 0, 0.5, 0],
            [0, 1.92, 0.5, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32), new_shape=self.shape)

        self.split_file, self.items_data = self.parse_items()
        self.should_resize = True  # Kitti always resizes due to different-shaped images.

    def log_args(self):
        self.logger.info(f"Split: '{self.split}' - Mode: '{self.mode}'")
        super().log_args()

    def validate_args(self) -> None:
        """Error checking for provided dataset configuration."""
        super().validate_args()

        if self.has('depth') and self.has('depth_velo'):
            raise ValueError('Must provide only one source of depth. (`depth`: Corrected LiDAR, `depth_velo`: LiDAR)')

    def parse_items(self) -> tuple[Path, ty.S[kr.Item]]:
        """Helper to parse dataset items."""
        file = kr.get_split_file(self.split, self.mode)
        lines = [line.split() for line in kr.load_split(file)]
        items = [kr.Item(line[0], int(line[1]), self._side2cam[line[2]]) for line in lines]
        return file, items

    def add_metadata(self, data: kr.Item, batch: ty.BatchData) -> ty.BatchData:
        """Add item metadata."""
        batch[2]['stem'] = f"{data.seq}/{data.cam}/{data.stem:010}"
        return batch

    def _load_image(self, data: kr.Item, offset: int = 0) -> Image:
        """Load target image from dataset. Offset should be used when loading support frames."""
        file = kr.get_image_file(data.seq, data.cam, data.stem+offset)
        if not file.is_file():
            exc = FileNotFoundError if offset == 0 else ty.SuppImageNotFoundError
            raise exc(f'Could not find specified file "{file}" with "{offset=}"')

        img = Image.open(file)
        if self.should_resize: img = img.resize(self.size, resample=Image.Resampling.BILINEAR)
        return img

    def get_supp_scale(self, data: kr.Item) -> int:
        """Generate the index of the support frame relative to the target image."""
        if not self.randomize_supp: return 1
        return 1

    def _load_stereo_image(self, data: kr.Item) -> Image:
        """Load the support stereo frame from dataset."""
        data = kr.Item(data.seq, data.stem, self._cam2stereo[data.cam])
        img = self._load_image(data)
        return img

    def _load_stereo_T(self, data: kr.Item) -> ty.A:
        """Load the stereo transform to the stereo frame from dataset."""
        T = np.eye(4, dtype=np.float32)
        T[0, 3] = self._cam2sign[data.cam] * 0.1  # Arbitrary baseline
        return T

    def _load_depth(self, data: kr.Item) -> ty.A:
        """Load ground-truth benchmark depth from dataset (corrected LiDAR)."""
        file = kr.get_depth_file(data.seq, data.cam, data.stem)
        if not file.is_file(): raise FileNotFoundError(f'Could not find specified depth benchmark file "{file}".')
        depth = kr.load_depth(file)
        if self.should_resize: depth = skit.resize(depth, self.SHAPE, order=0, preserve_range=True, mode='constant')
        return depth

    def load_depth_velo(self, data: kr.Item, batch: ty.BatchData) -> ty.BatchData:
        """Load ground-truth velodyne depth and store in loss targets. (y)"""
        batch[1]['depth'] = self._load_depth_velo(data)
        return batch

    def _load_depth_velo(self, data: kr.Item) -> ty.A:
        """Load ground-truth velodyne depth from dataset (raw LiDAR)."""
        file = kr.get_velodyne_file(data.seq, data.stem)
        if not file.is_file(): raise FileNotFoundError(f'Could not find specified depth LiDAR file "{file}".')

        seq = data.seq.split('/')[0]  # data['seq'] as 'sequence/drive'
        cam2cam, _, velo2cam = kr.load_calib(seq)
        depth = kr.load_depth_velodyne(file, velo2cam, cam2cam, cam=int(data.cam[-2:]))
        if self.should_resize: depth = skit.resize(depth, self.SHAPE, order=0, preserve_range=True, mode='constant')
        return depth

    def load_depth_hint(self, data: kr.Item, batch: ty.BatchData) -> ty.BatchData:
        """Load fused SGBM depth hints and store in loss targets. (y)"""
        batch[1]['depth_hints'] = self._load_depth_hint(data)
        return batch

    def _load_depth_hint(self, data: kr.Item) -> ty.A:
        """Load fused SGBM depth hints from dataset"""
        file = kr.get_hint_file(data.seq, data.cam, data.stem)
        if not file.is_file(): raise FileNotFoundError(f'Could not find specified depth hint file "{file}".')

        depth = np.load(file)
        if self.should_resize: depth = cv2.resize(depth, dsize=self.size, interpolation=cv2.INTER_NEAREST)
        return depth[..., None]

    def _load_K(self, data: kr.Item) -> ty.A:
        """Load camera intrinsics from dataset."""
        return self.K

    def apply_flip_aug(self, batch: ty.BatchData) -> ty.BatchData:
        """Apply horizontal flipping augmentation."""
        batch = super().apply_flip_aug(batch)
        y = batch[1]
        if 'depth_hints' in y: y['depth_hints'] = self.flip(y['depth_hints'])
        return batch

    def create_axs(self) -> ty.Axes:
        """Create axes required for displaying."""
        n = 1 + len(self.supp_idxs) + (self.has('depth') or self.has('depth_velo')) + self.has('depth_hint')
        _, axs = plt.subplots(n)
        if isinstance(axs, plt.Axes): axs = np.array([axs])
        plt.tight_layout()
        return axs

    def show(self, batch: ty.BatchData, axs: ty.Axes) -> None:
        """Show a single dataset item."""
        super().show(batch, axs)
        y = batch[1]
        if 'depth_hints' in y: axs[-1].imshow(viz.rgb_from_disp(y['depth_hints'], invert=True))


if __name__ == '__main__':
    ds = KittiRawDataset(
        split='eigen_zhou', mode='test', shape=(192, 640), datum=['image', 'support', 'depth', 'K'],
        as_torch=False, log_time=True, randomize=False, max_len=None, use_aug=True,
    )
    print(ds)
    ds.play(fps=1)

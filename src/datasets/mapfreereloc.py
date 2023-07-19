import random
from pathlib import Path

import skimage.transform as skit
from PIL import Image

import src.devkits.mapfreereloc as mfr
import src.typing as ty
from src import register
from src.tools import geometry as geo
from . import MdeBaseDataset

__all__ = ['MapFreeRelocDataset']


@register('mapfree')
class MapFreeRelocDataset(MdeBaseDataset):
    """MapFreeReloc dataset.

    Datum:
        - Image: Target image from which to predict depth.
        - Support: Adjacent frames (monocular) used to compute photometric consistency losses.
        - Pose: Camera extrinsic parameters.
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
            T: (Tensor) (4, 4) Camera extrinsics.
            K: (Tensor) (4, 4) Camera intrinsics.
        }

        m: {
            stem: (str) Path to loaded item.
            supp: (str) Support frame multiplier.
        }

    Parameters:
    :param mode: (str) Training mode to use. {train, val, test}

    Attributes:
    :attr mode: (str) Training mode used.
    :attr split_file: (Path) File containing the list of items in the loaded split.
    :attr items_data: (list[mfr.Item]) List of dataset items as (scene, seq, stem).
    """
    VALID_DATUM = 'image support depth pose K'
    SHAPE = 720, 540

    def __init__(self, mode: str, depth_src: str = 'dptkitti', **kwargs):
        super().__init__(**kwargs)
        self.mode = mode
        self.depth_src = depth_src
        self.split_file, self.items_data = self.parse_items()

    def log_args(self):
        self.logger.info(f"Mode: '{self.mode}'")
        super().log_args()

    def validate_args(self) -> None:
        super().validate_args()

        if 'depth' in self.datum and self.mode == 'train':
            raise ValueError('Depth maps are not available for the MapFreeReloc training split.')

        if self.supp_idxs and 0 in self.supp_idxs:
            raise ValueError('Stereo support frames are not provided by MapFreeReloc.')

    def parse_items(self) -> tuple[Path, ty.S[mfr.Item]]:
        file = mfr.Item.get_split_file(self.mode)
        data = mfr.Item.load_split(self.mode)
        return file, data

    def get_supp_scale(self, data: mfr.Item) -> int:
        if not self.randomize_supp: return 1
        k = random.randint(1, 5)
        return k

    def add_metadata(self, data: mfr.Item, batch: ty.BatchData) -> ty.BatchData:
        batch[2]['mode'] = self.mode
        batch[2]['scene'] = data.scene
        batch[2]['seq'] = data.seq
        batch[2]['stem'] = data.stem
        return batch

    def load_pose(self, data: mfr.Item, batch: ty.BatchData) -> ty.BatchData:
        batch[1]['T'] = self._load_pose(data)
        return batch

    def _load_image(self, data: mfr.Item, offset: int = 0) -> Image:
        if offset:
            stem = int(data.stem.split('_')[1]) + offset
            data = mfr.Item(self.mode, data.scene, data.seq, f'frame_{stem:05d}')

        if not data.get_img_file().is_file():
            exc = FileNotFoundError if offset == 0 else ty.SuppImageNotFoundError
            raise exc(f'Could not find specified file "{data.scene}/{data.seq}/{data.stem}" with "{offset=}"')

        img = data.load_img()
        if self.should_resize: img = img.resize(self.size, resample=Image.Resampling.BILINEAR)
        return img

    def _load_depth(self, data: mfr.Item) -> ty.A:
        depth = data.load_depth(self.depth_src)
        if self.should_resize: skit.resize(depth, self.SHAPE, order=0, preserve_range=True, mode='constant')
        return depth

    def _load_pose(self, data: mfr.Item) -> ty.A:
        return data.load_pose()

    def _load_K(self, data: mfr.Item) -> ty.A:
        K = data.load_intrinsics()
        if self.should_resize: K = geo.resize_K(K, self.shape, self.SHAPE)
        return K

    def _load_stereo_image(self, data: mfr.Item) -> Image:
        raise NotImplementedError('MapFreeReloc does not provide stereo images.')

    def _load_stereo_T(self, data: mfr.Item) -> ty.A:
        raise NotImplementedError('MapFreeReloc does not provide stereo images.')


if __name__ == '__main__':
    ds = MapFreeRelocDataset(mode='val', datum='image depth', shape=None, supp_idxs=None, as_torch=False)
    print(len(ds))
    ds.play(fps=1)

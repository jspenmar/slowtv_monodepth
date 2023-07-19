import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import src.devkits.mannequin as mc
import src.typing as ty
from src import register
from src.tools import geometry as geo
from . import MdeBaseDataset

__all__ = ['MannequinDataset']


@register('mannequin')
class MannequinDataset(MdeBaseDataset):
    """Mannequin Challenge dataset.

   Datum:
       - Image: Target image from which to predict depth.
       - Support: Adjacent frames (monocular) used to compute photometric consistency losses.
       - Depth: Target ground-truth COLMAP depth.
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
           supp_imgs: (Tensor) (n, 3, h, w) Augmented support frames.
           depth: (Tensor) (1, h, w) Ground-truth target depth.
           K: (Tensor) (4, 4) Camera intrinsics.
       }

       m: {
           supp: (str) Support frame multiplier.
       }

   Parameters:
   :param mode: (str) Training mode to use. {train, val, test}

   Attributes:
   :attr split_file: (Path) File containing the list of items in the loaded split.
   :attr items_data: (list[mc.Item]) List of dataset items as (seq, cam, stem).
   """
    VALID_DATUM = 'image support depth K'
    SHAPE = 1080, 1920

    def __init__(self, mode: str, **kwargs):
        super().__init__(**kwargs)
        self.mode = mode
        self.split_file, self.items_data = self.parse_items()

    def log_args(self):
        self.logger.info(f"Mode: '{self.mode}'")
        super().log_args()

    def validate_args(self) -> None:
        """Error checking for provided dataset configuration."""
        super().validate_args()
        if 0 in self.supp_idxs: raise ValueError('MannequinChallenge does not provide stereo pairs.')

    def parse_items(self) -> tuple[Path, ty.S[mc.Item]]:
        """Helper to parse dataset items."""
        return mc.load_split(self.mode)

    def add_metadata(self, data: mc.Item, batch: ty.BatchData) -> ty.BatchData:
        batch[2]['seq'] = data.seq
        return batch

    def _load_image(self, data: mc.Item, offset: int = 0) -> Image:
        """Load target image from dataset. Offset should be used when loading support frames."""
        file = mc.get_img_file(mode=self.mode, seq=data.seq, stem=int(data[1]) + offset)
        if not file.is_file():
            exc = FileNotFoundError if offset == 0 else ty.SuppImageNotFoundError
            raise exc(f'Could not find specified file "{file}" with "{offset=}"')

        img = Image.open(file)
        if self.should_resize: img = img.resize(self.size, resample=Image.Resampling.BILINEAR)
        return img

    def get_supp_scale(self, data: mc.Item) -> int:
        """Generate the index of the support frame relative to the target image."""
        if not self.randomize_supp: return 1
        k = random.randint(1, 5)
        return k

    def _load_depth(self, data: mc.Item) -> None:
        """Load ground-truth depth from dataset."""
        file = mc.get_depth_file(self.mode, data.seq, data.stem)
        if not file.is_file(): raise FileNotFoundError(f'Could not find specified depth file "{file}".')

        depth = np.load(file)
        if self.should_resize: depth = cv2.resize(depth, dsize=self.size, interpolation=cv2.INTER_NEAREST)
        return depth

    def _load_K(self, data: mc.Item) -> ty.A:
        """Load camera intrinsics from dataset."""
        info = mc.load_info(self.mode, data.seq)[data.stem]
        sh = [i+1 for i in info['shape']]
        K = info['K']
        if self.should_resize: K = geo.resize_K(K, self.shape, shape=sh)
        return K

    def _load_stereo_image(self, data: mc.Item) -> None:
        raise NotImplementedError('MannequinChallenge does not contain stereo pairs.')

    def _load_stereo_T(self, data: mc.Item) -> None:
        raise NotImplementedError('MannequinChallenge does not contain stereo pairs.')


if __name__ == '__main__':
    ds = MannequinDataset(
        mode='test', shape=(384, 640), datum=('image', 'depth', 'K'),
        supp_idxs=None, randomize_supp=False,
        as_torch=False, use_aug=False, log_time=False, max_len=None, randomize=True,
    )
    print(ds)
    ds.play(fps=1)

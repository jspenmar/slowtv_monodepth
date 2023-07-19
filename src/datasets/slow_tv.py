import random
from pathlib import Path

from PIL import Image

import src.devkits.slow_tv as stv
import src.typing as ty
from src import register
from src.tools import geometry as geo
from . import MdeBaseDataset

__all__ = ['SlowTvDataset']


@register('slow_tv')
class SlowTvDataset(MdeBaseDataset):
    """SlowTV dataset.

    Datum:
        - Image: Target image from which to predict depth.
        - Support: Adjacent frames (monocular) used to compute photometric consistency losses.
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
            K: (Tensor) (4, 4) Camera intrinsics.
        }

        m: {
            supp: (str) Support frame multiplier.
        }

    Parameters:
    :param split: (str) SlowTV split to use. {all, natural, driving, underwater}
    :param mode: (str) Training mode to use. {train, val}

    Attributes:
    :attr split_file: (Path) File containing the list of items in the loaded split.
    :attr items_data: (list[stv.Item]) List of dataset items as (seq, stem).
    :attr cats: (dict[str, str]) Dict containing the category of each sequence.
    """
    VALID_DATUM = 'image support K'
    SHAPE = 720, 1280

    def __init__(self, split: str, mode: str, **kwargs):
        super().__init__(**kwargs)
        self.split = split
        self.mode = mode

        self.split_file, self.items_data = self.parse_items()
        self.cats = self.parse_cats()

        self._max_offset_per_cat = {
            'natural': 5,
            'driving': 1,
            'underwater': 5,
        }

    def log_args(self):
        self.logger.info(f"Split: '{self.split}' - Mode: '{self.mode}'")
        super().log_args()

    def validate_args(self) -> None:
        """Error checking for provided dataset configuration."""
        super().validate_args()
        if 0 in self.supp_idxs: raise ValueError('SlowTV does not provide stereo pairs.')

    def parse_items(self) -> tuple[Path, ty.S[stv.Item]]:
        """Helper to parse dataset items."""
        file, items = stv.load_split(self.mode, self.split)
        return file, items

    def parse_cats(self) -> dict[str, str]:
        """Helper to load the category for each sequence."""
        return {seq: c for seq, c in zip(stv.get_seqs(), stv.load_categories(subcats=False))}

    def _load_image(self, data: stv.Item, offset: int = 0) -> Image:
        """Load target image from dataset. Offset should be used when loading support frames."""
        file = stv.get_img_file(seq=data.seq, stem=int(data.stem) + offset)
        if not file.is_file():
            exc = FileNotFoundError if offset == 0 else ty.SuppImageNotFoundError
            raise exc(f'Could not find specified file "{file}" with "{offset=}"')

        img = Image.open(file)
        if self.should_resize: img = img.resize(self.size, resample=Image.Resampling.BILINEAR)
        return img

    def get_supp_scale(self, data: stv.Item) -> int:
        """Generate the index of the support frame relative to the target image."""
        if not self.randomize_supp: return 1

        cat = self.cats[data.seq]
        k = random.randint(1, self._max_offset_per_cat[cat])
        return k

    def _load_K(self, data: stv.Item) -> ty.A:
        """Load camera intrinsics from dataset."""
        K = stv.load_intrinsics(data[0])
        if self.should_resize: K = geo.resize_K(K, self.shape, self.SHAPE)
        return K

    def _load_stereo_image(self, data: stv.Item) -> None:
        raise NotImplementedError('SlowTV does not contain stereo pairs.')

    def _load_stereo_T(self, data: stv.Item) -> None:
        raise NotImplementedError('SlowTV does not contain stereo pairs.')

    def _load_depth(self, data: stv.Item) -> None:
        raise NotImplementedError('SlowTV does not contain ground-truth depth.')


if __name__ == '__main__':

    ds = SlowTvDataset(
        split='00039', mode='train', shape=(384, 640), datum=('image', 'support', 'K'),
        supp_idxs=[1, -1], randomize_supp=False,
        as_torch=False, use_aug=True, log_time=False,
    )
    ds.play(fps=1, skip=1, reverse=False)

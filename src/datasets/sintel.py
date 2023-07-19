from pathlib import Path

from PIL import Image

import src.devkits.sintel as si
import src.typing as ty
from src import register
from src.tools import geometry as geo
from . import MdeBaseDataset

__all__ = ['SintelDataset']


@register('sintel')
class SintelDataset(MdeBaseDataset):
    VALID_DATUM = 'image depth K'
    SHAPE = 436, 1024

    def __init__(self,
                 mode: str,
                 datum: ty.U[str, ty.S[str]] = 'image depth K',
                 **kwargs):
        super().__init__(datum=datum, **kwargs)
        self.mode = mode
        self.split_file, self.items_data = self.parse_items()

    def log_args(self):
        self.logger.info(f"Mode: '{self.mode}'")
        super().log_args()

    def validate_args(self) -> None:
        self.VALID_DATUM.add('support')
        super().validate_args()
        self.VALID_DATUM.remove('support')

        if self.use_aug: raise ValueError('Sintel is a testing dataset, no augmentations should be applied.')
        if self.supp_idxs: raise ValueError('Sintel does not provide support frames.')
        if self.randomize_supp: raise ValueError('Sintel does not provide support frames.')

    def parse_items(self) -> tuple[Path, ty.S[si.Item]]:
        file = si.Item.get_split_file(self.mode)
        data = si.Item.load_split(self.mode)
        return file, data

    def _load_image(self, data: si.Item, offset: int = 0) -> Image:
        img = data.load_img()
        if self.should_resize: img = img.resize(self.size, resample=Image.Resampling.BILINEAR)
        return img

    def _load_depth(self, data: si.Item) -> ty.A:
        return data.load_depth()

    def _load_K(self, data: si.Item) -> ty.A:
        K = data.load_intrinsics()
        if self.should_resize: K = geo.resize_K(K, self.shape, self.SHAPE)
        return K

    def get_supp_scale(self, data: si.Item) -> ty.A:
        raise NotImplementedError('Sintel does not provide support frames.')

    def _load_stereo_image(self, data: si.Item) -> ty.A:
        raise NotImplementedError('Sintel does not provide stereo frames.')

    def _load_stereo_T(self, data: si.Item) -> ty.A:
        raise NotImplementedError('Sintel does not provide stereo frames.')


if __name__ == '__main__':
    ds = SintelDataset(mode='train', as_torch=False, max_len=None, randomize=False)
    ds.play(fps=1)

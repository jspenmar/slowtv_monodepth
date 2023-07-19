from pathlib import Path

from PIL import Image

import src.typing as ty
from src import register
from src.devkits import nyud
from . import MdeBaseDataset

__all__ = ['NyudDataset']


@register('nyud')
class NyudDataset(MdeBaseDataset):
    VALID_DATUM = 'image depth'
    SHAPE = 480, 640

    def __init__(self,
                 mode: str,
                 datum: ty.U[str, ty.S[str]] = 'image depth',
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

        if self.use_aug: raise ValueError('NYUD-v2 is a testing dataset, no augmentations should be applied.')
        if self.supp_idxs: raise ValueError('NYUD-v2 does not provide support frames.')
        if self.randomize_supp: raise ValueError('NYUD-v2 does not provide support frames.')

    def parse_items(self) -> tuple[Path, ty.S[nyud.Item]]:
        file = nyud.Item.get_split_file(self.mode)
        data = nyud.Item.load_split(self.mode)
        return file, data

    def _load_image(self, data: nyud.Item, offset: int = 0) -> Image:
        img = data.load_img()
        if self.should_resize: img = img.resize(self.size, resample=Image.Resampling.BILINEAR)
        return img

    def _load_depth(self, data: nyud.Item) -> ty.A:
        return data.load_depth()

    def get_supp_scale(self, data: nyud.Item) -> ty.A:
        raise NotImplementedError('NYUD-v2 does not provide support frames.')

    def _load_K(self, data: nyud.Item) -> ty.A:
        raise NotImplementedError('NYUD-v2 does not provide camera intrinsics.')

    def _load_stereo_image(self, data: nyud.Item) -> ty.A:
        raise NotImplementedError('NYUD-v2 does not provide stereo frames.')

    def _load_stereo_T(self, data: nyud.Item) -> ty.A:
        raise NotImplementedError('NYUD-v2 does not provide stereo frames.')


if __name__ == '__main__':
    ds = NyudDataset(mode='test', as_torch=False, max_len=None, randomize=True)
    ds.play(fps=1)

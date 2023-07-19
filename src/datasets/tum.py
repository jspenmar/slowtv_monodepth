from pathlib import Path

from PIL import Image

import src.typing as ty
from src import register
from src.devkits import tum
from . import MdeBaseDataset

__all__ = ['TumDataset']


@register('tum')
class TumDataset(MdeBaseDataset):
    VALID_DATUM = 'image depth'
    SHAPE = 480, 640

    def __init__(self,
                 mode: str,
                 datum='image depth',
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

        if self.use_aug: raise ValueError('TUM-RGBD is a testing dataset, no augmentations should be applied.')
        if self.supp_idxs: raise ValueError('TUM-RGBD does not provide support frames.')
        if self.randomize_supp: raise ValueError('TUM-RGBD does not provide support frames.')

    def parse_items(self) -> tuple[Path, ty.S[tum.Item]]:
        file = tum.Item.get_split_file(self.mode)
        data = tum.Item.load_split(self.mode)
        return file, data

    def _load_image(self, data: tum.Item, offset: int = 0) -> Image:
        img = data.load_img()
        if self.should_resize: img = img.resize(self.size, resample=Image.Resampling.BILINEAR)
        return img

    def _load_depth(self, data: tum.Item) -> ty.A:
        return data.load_depth()

    def get_supp_scale(self, data: tum.Item) -> ty.A:
        raise NotImplementedError('TUM-RGBD does not provide support frames.')

    def _load_K(self, data: tum.Item) -> ty.A:
        raise NotImplementedError('TUM-RGBD does not provide camera intrinsics.')

    def _load_stereo_image(self, data: tum.Item) -> ty.A:
        raise NotImplementedError('TUM-RGBD does not provide stereo frames.')

    def _load_stereo_T(self, data: tum.Item) -> ty.A:
        raise NotImplementedError('TUM-RGBD does not provide stereo frames.')


if __name__ == '__main__':
    ds = TumDataset(mode='test', as_torch=False, max_len=None, randomize=False)
    ds.play(fps=1, skip=1, reverse=True)

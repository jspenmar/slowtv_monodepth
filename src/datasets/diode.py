from pathlib import Path

from PIL import Image

import src.devkits.diode as di
import src.typing as ty
from src import register
from . import MdeBaseDataset

__all__ = ['DiodeDataset']


@register('diode')
class DiodeDataset(MdeBaseDataset):
    VALID_DATUM = 'image depth mask'
    SHAPE = 768, 1024

    def __init__(self,
                 scene: str,
                 mode: str,
                 datum='image depth mask',
                 **kwargs):
        super().__init__(datum=datum, **kwargs)
        self.scene = scene
        self.mode = mode
        self.split_file, self.items_data = self.parse_items()

    def log_args(self):
        self.logger.info(f"Split: '{self.scene}' - Mode: '{self.mode}'")
        super().log_args()
        
    def validate_args(self) -> None:
        self.VALID_DATUM.add('support')
        super().validate_args()
        self.VALID_DATUM.remove('support')

        if self.use_aug: raise ValueError('Diode is a testing dataset, no augmentations should be applied.')
        if self.supp_idxs: raise ValueError('Diode does not provide support frames.')
        if self.randomize_supp: raise ValueError('Diode does not provide support frames.')

    def parse_items(self) -> tuple[Path, ty.S[di.Item]]:
        file = di.Item.get_split_file(self.mode, self.scene)
        data = di.Item.load_split(self.mode, self.scene)
        return file, data

    def _load_image(self, data: di.Item, offset: int = 0) -> Image:
        img = data.load_img()
        if self.should_resize: img = img.resize(self.size, resample=Image.Resampling.BILINEAR)
        return img

    def _load_depth(self, data: di.Item) -> ty.A:
        return data.load_depth()

    def load_mask(self, data: di.Item, batch: ty.BatchData) -> ty.BatchData:
        batch[1]['mask'] = self._load_mask(data)
        return batch

    def _load_mask(self, data: di.Item) -> ty.A:
        return data.load_mask()

    def get_supp_scale(self, data: di.Item) -> ty.A:
        raise NotImplementedError('Diode does not provide support frames.')

    def _load_K(self, data: di.Item) -> ty.A:
        raise NotImplementedError('Diode does not provide camera intrinsics.')

    def _load_stereo_image(self, data: di.Item) -> ty.A:
        raise NotImplementedError('Diode does not provide stereo frames.')

    def _load_stereo_T(self, data: di.Item) -> ty.A:
        raise NotImplementedError('Diode does not provide stereo frames.')


if __name__ == '__main__':
    ds = DiodeDataset(mode='val', scene='indoors', as_torch=False, max_len=None, randomize=True)
    ds.play(fps=1)

from pathlib import Path

import cv2
from PIL import Image

import src.devkits.mannequin_lmdb as mc
import src.typing as ty
from src import register
from src.tools import geometry as geo
from . import MannequinDataset

__all__ = ['MannequinLmdbDataset']


@register('mannequin_lmdb')
class MannequinLmdbDataset(MannequinDataset):
    """Mannequin Challenge dataset using LMDBs. See `MannequinDataset` for additional details."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.img_db = mc.load_imgs(self.mode)
        self.depth_db = mc.load_depths(self.mode) if self.has('depth') else None
        self.K_db = mc.load_intrinsics(self.mode)
        self.shape_db = mc.load_shapes(self.mode)

    def parse_items(self) -> tuple[Path, ty.S[mc.Item]]:
        """Helper to parse dataset items."""
        return mc.load_split(self.mode)

    def _load_image(self, data: mc.Item, offset: int = 0) -> Image:
        """Load target image from dataset. Offset should be used when loading support frames."""
        k = f'{data.seq}/{int(data.stem)+offset:05}'

        if k not in self.img_db:
            exc = FileNotFoundError if offset == 0 else ty.SuppImageNotFoundError
            raise exc(f'Could not find specified file "{k}" with "{offset=}"')

        img = self.img_db[k]
        if self.should_resize: img = img.resize(self.size, resample=Image.Resampling.BILINEAR)
        return img

    def _load_depth(self, data: mc.Item) -> None:
        """Load ground-truth depth from dataset."""
        k = f'{data.seq}/{data.stem}'
        if k not in self.img_db: raise FileNotFoundError(f'Could not find specified file "{k}"')

        depth = self.depth_db[k]
        if self.should_resize: depth = cv2.resize(depth, dsize=self.size, interpolation=cv2.INTER_NEAREST)
        return depth

    def _load_K(self, data: mc.Item) -> ty.A:
        """Load camera intrinsics from dataset."""
        K = self.K_db[data.seq]
        if self.should_resize: K = geo.resize_K(K, self.shape, shape=self._load_shape(data))
        return K

    def _load_shape(self, data: mc.Item) -> ty.S[int]:
        """Load original image shape from dataset."""
        sh = self.shape_db[data.seq]
        sh = [i+1 for i in sh]
        return sh


if __name__ == '__main__':
    ds = MannequinLmdbDataset(
        mode='train', shape=(384, 640), datum=('image', 'support', 'K'),
        supp_idxs=[-1, 1], randomize_supp=False,
        as_torch=False, use_aug=False, log_time=False,
    )
    print(ds)
    ds.play(fps=1)

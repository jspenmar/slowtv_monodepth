from pathlib import Path

from PIL import Image

import src.devkits.slow_tv_lmdb as stv
import src.typing as ty
from src import register
from src.tools import geometry as geo
from .slow_tv import SlowTvDataset

__all__ = ['SlowTvLmdbDataset']


@register('slow_tv_lmdb')
class SlowTvLmdbDataset(SlowTvDataset):
    """SlowTV dataset using LMDBs. See `SlowTvDataset` for additional details."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_dbs = {}
        self.calib_db = stv.load_calibs()
        self.preload()

    def parse_items(self) -> tuple[Path, ty.S[stv.Item]]:
        """Helper to parse dataset items."""
        return stv.load_split(self.mode, self.split)

    def parse_cats(self) -> dict[str, str]:
        """Helper to load the category for each sequence."""
        return {seq: c for seq, c in zip(stv.get_seqs(), stv.load_categories(subcats=False))}

    def preload(self) -> None:
        """Create all LMDBs for the required items."""
        seqs = set(i.seq for i in self.items_data)
        for s in seqs: self.image_dbs[s] = stv.load_imgs(s)

    def _load_image(self, data: stv.Item, offset: int = 0) -> Image:
        """Load target image from dataset. Offset should be used when loading support frames."""
        k = f'{int(data.stem) + offset:010}'
        kdb = data.seq

        db = self.image_dbs[kdb]
        if k not in db:
            exc = FileNotFoundError if offset == 0 else ty.SuppImageNotFoundError
            raise exc(f'Could not find specified file "{kdb}/{k}" with "{offset=}"')

        img = db[k]
        if self.should_resize: img = img.resize(self.size, resample=Image.Resampling.BILINEAR)
        return img

    def _load_K(self, data: stv.Item) -> ty.A:
        """Load camera intrinsics from dataset."""
        K = self.calib_db[data.seq]
        if self.should_resize: K = geo.resize_K(K, self.shape, self.SHAPE)
        return K


if __name__ == '__main__':

    ds = SlowTvLmdbDataset(
        split='00001', mode='train', shape=(384, 640), datum=('image', 'support', 'K'),
        supp_idxs=[1, -1], randomize_supp=False,
        as_torch=False, use_aug=True, log_time=False,
    )
    ds.play(fps=1, skip=1, reverse=False)

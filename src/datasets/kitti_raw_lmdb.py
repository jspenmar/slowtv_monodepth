from pathlib import Path

import cv2
import skimage.transform as skit
from PIL import Image

import src.devkits.kitti_raw_lmdb as kr
import src.typing as ty
from src import register
from .kitti_raw import KittiRawDataset

__all__ = ['KittiRawLmdbDataset']


@register('kitti_lmdb')
class KittiRawLmdbDataset(KittiRawDataset):
    """Kitti Raw dataset using LMDBs. See `KittiRawDataset` for additional details."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_dbs = {}
        self.depth_dbs = {}
        self.poses_dbs = {}
        self.hints_dbs = {}
        self.calib_dbs = {}

        self.preload()

    def parse_items(self) -> tuple[Path, ty.S[kr.Item]]:
        """Helper to parse dataset items."""
        file = kr.get_split_file(self.split, self.mode)
        lines = [line.split() for line in kr.load_split(file)]
        items = [kr.Item(line[0], int(line[1]), self._side2cam[line[2]]) for line in lines]
        return file, items

    def preload(self) -> None:
        """Create all LMDBs for the required items."""
        drives = set(item.seq for item in self.items_data)

        for d in drives:
            self.image_dbs[f'{d}/image_02'] = kr.load_images(*d.split('/'), 'image_02')
            self.image_dbs[f'{d}/image_03'] = kr.load_images(*d.split('/'), 'image_03')

        if self.has('depth'):
            for d in drives:
                self.depth_dbs[f'{d}/image_02'] = kr.load_depths(*d.split('/'), 'image_02')
                self.depth_dbs[f'{d}/image_03'] = kr.load_depths(*d.split('/'), 'image_03')

        if self.has('depth_velo'):
            seqs = set(seq.split('/')[0] for seq in drives)
            self.calib_dbs = {s: kr.load_calib(s) for s in seqs}

            for d in drives:
                seq, drive = d.split('/')
                self.depth_dbs[d] = kr.load_velo_depths(seq, drive, self.calib_dbs[seq])

        if self.has('depth_hint'):
            for d in drives:
                self.hints_dbs[f'{d}/image_02'] = kr.load_hints(*d.split('/'), 'image_02')
                self.hints_dbs[f'{d}/image_03'] = kr.load_hints(*d.split('/'), 'image_03')

    def _load_image(self, data: kr.Item, offset: int = 0) -> Image:
        """Load target image from dataset. Offset should be used when loading support frames."""
        k = f'{data.stem+offset:010}'
        kdb = f'{data.seq}/{data.cam}'

        db = self.image_dbs[kdb]
        if k not in db:
            exc = FileNotFoundError if offset == 0 else ty.SuppImageNotFoundError
            raise exc(f'Could not find specified file "{kdb}/{k}" with "{offset=}"')

        img = db[k]
        if self.should_resize: img = img.resize(self.size, resample=Image.Resampling.BILINEAR)
        return img

    def _load_depth(self, data: kr.Item) -> ty.A:
        """Load ground-truth benchmark depth from dataset (corrected LiDAR)."""
        k = f'{data.stem:010}'
        kdb = f'{data.seq}/{data.cam}'
        depth = self.depth_dbs[kdb][k]
        if self.should_resize: depth = skit.resize(depth, self.SHAPE, order=0, preserve_range=True, mode='constant')
        return depth[..., None]

    def _load_depth_velo(self, data: kr.Item) -> ty.A:
        """Load ground-truth velodyne depth from dataset (raw LiDAR)."""
        k = (f'{data.stem:010}', int(data.cam[-2:]))
        kdb = data.seq

        depth = self.depth_dbs[kdb][k]
        if self.should_resize: depth = skit.resize(depth, self.SHAPE, order=0, preserve_range=True, mode='constant')
        return depth[..., None]

    def _load_depth_hint(self, data: kr.Item) -> ty.A:
        """Load fused SGBM depth hints and store in loss targets. (y)"""
        k = f'{data.stem:010}'
        kdb = f'{data.seq}/{data.cam}'
        depth = self.hints_dbs[kdb][k]
        if self.should_resize: depth = cv2.resize(depth, dsize=self.size, interpolation=cv2.INTER_NEAREST)
        return depth[..., None]


if __name__ == '__main__':
    ds = KittiRawLmdbDataset(
        split='eigen_zhou', mode='test', shape=(192, 640), datum=['image', 'support', 'depth', 'depth_hint', 'K'],
        as_torch=False
    )
    ds.play(fps=1)

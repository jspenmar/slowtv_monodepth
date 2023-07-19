from pathlib import Path

from PIL import Image

import src.typing as ty
from src import register
from src.devkits import ddad
from src.tools import geometry as geo
from . import MdeBaseDataset

__all__ = ['DdadDataset']


@register('ddad')
class DdadDataset(MdeBaseDataset):
    """DDAD Dataset. From: https://arxiv.org/abs/1905.02693.

    This dataset is a simple wrapper over the official `SynchronizedSceneDataset` provided by the DGP repo
    (https://github.com/TRI-ML/dgp, downloaded to `/PATH/TO/ROOT/src/external_libs/dgp`).

    The current implementation focuses exclusively on using DDAD as a testing dataset and should not be used for
    training. By default, we also limit the dataset to the first 1000 items.

    Datum:
        image: Target image from which to predict depth.
        depth: Target ground-truth Lidar depth.
        K: Camera intrinsic parameters.

    Batch:
        x: {
            imgs: (Tensor) (3, h, w) Augmented target image.
        }

        y: {
            imgs: (Tensor) (3, h, w) Non-augmented target image.
            depth: (Tensor) (1, h, w) Ground-truth target depth (either Benchmark or LiDAR)
            K: (Tensor) (4, 4) Camera intrinsics.
        }

        m: {}

    Parameters:
    :param mode: (str) Dataset split mode to load. {val}
    :param kwargs: (dict) Kwargs accepted by `BaseDataset` and `MdeBaseDataset`.
    """
    VALID_DATUM = 'image depth K'
    SHAPE = 1216, 1936

    def __init__(self, mode: str, **kwargs):
        super().__init__(**kwargs)
        self.mode = mode
        self.max_len = 1000
        self.split_file, self.items_data = self.parse_items()

    def log_args(self) -> None:
        self.logger.info(f"Mode: '{self.mode}'")
        super().log_args()

    def validate_args(self) -> None:
        self.VALID_DATUM.add('support')
        super().validate_args()
        self.VALID_DATUM.remove('support')

        if self.mode != 'val': raise ValueError("DDAD is a testing dataset. Only a `val` split is provided.")
        if self.use_aug: raise ValueError("DDAD is a testing dataset, no augmentations should be applied.")
        if self.supp_idxs: raise ValueError("DDAD does not provide support frames.")
        if self.randomize_supp: raise ValueError("DDAD does not provide support frames.")

    def parse_items(self) -> tuple[Path, ddad.SynchronizedSceneDataset]:
        file = ddad.get_json_file()
        datum = ['camera_01'] + (['lidar'] if 'depth' in self.datum else [])
        ds = ddad.get_dataset(self.mode, datum=datum)
        return file, ds

    def _load_image(self, data: ty.Any, offset: int = 0) -> Image:
        img = data[0][0]['rgb']
        if self.should_resize: img = img.resize(self.size, resample=Image.Resampling.BILINEAR)
        return img

    def _load_depth(self, data: ty.Any) -> ty.A:
        depth = data[0][0]['depth']
        return depth

    def _load_K(self, data: ty.Any) -> ty.A:
        K = geo.pad_K(data[0][0]['intrinsics'])
        if self.should_resize: K = geo.resize_K(K, self.shape, self.SHAPE)
        return K

    def get_supp_scale(self, data: ty.Any) -> ty.A:
        raise NotImplementedError("DDAD does not provide support frames.")

    def _load_stereo_image(self, data: ty.Any) -> ty.A:
        raise NotImplementedError("DDAD does not provide stereo frames.")

    def _load_stereo_T(self, data: ty.Any) -> ty.A:
        raise NotImplementedError("DDAD does not provide stereo frames.")


if __name__ == '__main__':
    ds = DdadDataset(mode='val', datum='image depth K', as_torch=False, max_len=10, randomize=True)
    ds.play(fps=1, skip=1, reverse=False)

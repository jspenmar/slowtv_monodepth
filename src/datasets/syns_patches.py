import numpy as np
import skimage.transform as skit
from PIL import Image
from matplotlib import pyplot as plt

import src.devkits.syns_patches as syp
import src.typing as ty
from src import register
from src.tools import geometry as geo
from src.utils import io
from . import MdeBaseDataset

__all__ = ['SynsPatchesDataset']


@register('syns_patches')
class SynsPatchesDataset(MdeBaseDataset):
    """SYNS-Patches dataset.

    Datum:
        - Image: Target image from which to predict depth.
        - Depth: Target ground-truth depth.
        - Edge: Target ground-truth depth boundaries.
        - K: Camera intrinsic parameters.

    See BaseDataset for additional added metadata.

    Batch:
        x: {
            imgs: (Tensor) (3, h, w) Augmented target image.
        }

        y: {
            imgs: (Tensor) (3, h, w) Non-augmented target image.
            depth: (Tensor) (1, h, w) Ground-truth target depth.
            edges: (Tensor) (1, h, w) Ground-truth target depth boundaries.
            K: (Tensor) (4, 4) Camera intrinsics.
        }

        m: {
            cat: (str) Scene category.
            subcat: (str) Scene sub-category.
            supp: (str) Support frame multiplier.
        }

    Parameters:
    :param mode: (str) Training mode to use. {val, test}
    :param downsample_gt: (bool) If `True` downsample the gorund-truth depth to match the image resolution.

    Attributes:
    :attr split_file: (Path) File containing the list of items in the loaded split.
    :attr items_data: (list[syp.Item]) List of dataset items as (seq, cam, stem).
    """
    VALID_DATUM = 'image depth edge K'
    SHAPE = 376, 1242

    def __init__(self,
                 mode: str,
                 datum: ty.S[str] = 'image depth edge K',
                 **kwargs):
        super().__init__(datum=datum, **kwargs)
        self.mode = mode
        self.split_file, self.items_data = self.parse_items()

    def log_args(self):
        self.logger.info(f"Mode: '{self.mode}'")
        super().log_args()

    def validate_args(self) -> None:
        """Error checking for provided dataset configuration."""
        self.VALID_DATUM.add('support')  # Fake support during error checking
        super().validate_args()
        self.VALID_DATUM.remove('support')

        if self.use_aug: raise ValueError('SYNS-Patches is a testing dataset, no augmentations should be applied.')
        if self.supp_idxs: raise ValueError('SYNS-Patches does not provide support frames.')
        if self.randomize_supp: raise ValueError('SYNS-Patches does not provide support frames.')

    def parse_items(self):
        """Helper to parse dataset items."""
        return syp.load_split(self.mode)

    def add_metadata(self, data: syp.Item, batch: ty.BatchData) -> ty.BatchData:
        """Add item metadata."""
        m = batch[2]
        m['cat'], m['subcat'] = syp.load_category(data[0])
        return batch

    def _load_image(self, data: syp.Item, offset: int = 0) -> Image:
        """Load target image from dataset. Offset should be used when loading support frames."""
        file = syp.get_image_file(data.seq, data.stem)
        img = Image.open(file)
        if self.should_resize: img = img.resize(self.size, resample=Image.Resampling.BILINEAR)
        return img

    def _load_depth(self, data: syp.Item) -> None:
        """Load ground-truth benchmark depth from dataset (corrected LiDAR)."""
        file = syp.get_depth_file(data.seq, data.stem)
        depth = np.load(file).astype(np.float32)
        if self.should_resize: depth = skit.resize(depth, self.shape, order=0, preserve_range=True, mode='constant')
        return depth

    def load_edge(self, data: syp.Item, batch: ty.BatchData) -> ty.BatchData:
        """Load ground-truth depth boundaries and add to loss targets. (y)"""
        edges = self._load_edge(data)
        batch[1]['edges'] = io.pil2np(edges)[..., None].astype(bool)
        return batch

    def _load_edge(self, data: syp.Item) -> Image:
        """Load ground-truth depth boundaries from dataset."""
        file = syp.get_edges_file(data.seq, 'edges', data.stem)
        edge = Image.open(file)
        if self.should_resize: edge = edge.resize(self.size, resample=Image.Resampling.NEAREST)
        return edge

    def _load_K(self, data: syp.Item) -> ty.A:
        """Load camera intrinsics from dataset."""
        K = syp.load_intrinsics()
        if self.should_resize: K = geo.resize_K(K, self.shape, self.SHAPE)
        return K

    def create_axs(self) -> ty.Axes:
        """Create axes required for displaying."""
        _, axs = plt.subplots(1 + self.has('depth') + self.has('edge'))
        if isinstance(axs, plt.ty.Axes): axs = np.array([axs])
        plt.tight_layout()
        return axs

    def show(self, batch: ty.BatchData, axs: ty.Axes) -> None:
        """Show a single dataset item."""
        super().show(batch, axs)
        y = batch[1]
        if 'edges' in y: axs[-1].imshow(y['edges'])

    def get_supp_scale(self, data: syp.Item) -> int:
        raise NotImplementedError('SYNS-Patches does not contain support frames.')

    def _load_stereo_image(self, data: syp.Item) -> None:
        raise NotImplementedError('SYNS-Patches does not contain stereo pairs.')

    def _load_stereo_T(self, data: syp.Item) -> None:
        raise NotImplementedError('SYNS-Patches does not contain stereo pairs.')

    def load_support(self, data: syp.Item, batch: ty.BatchData) -> ty.BatchData:
        raise NotImplementedError('SYNS-Patches does not contain support frames.')


if __name__ == '__main__':
    ds = SynsPatchesDataset(
        mode='test', shape=(192, 640), datum=('image', 'depth', 'edge', 'K'),
        as_torch=False, use_aug=False, randomize=False, max_len=None,
    )
    print(ds)
    ds.play(1)

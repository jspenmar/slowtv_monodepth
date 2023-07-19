import random
from functools import partial

import kornia.geometry.transform as KT
import torch
import torch.nn.functional as F

import src.typing as ty
from src.tools import geometry as geo

__all__ = ['aspect_ratio_aug']


LABELS = [
    '6/13', '9/16', '3/5', '2/3', '4/5', '1/1',  # Portrait
    '5/4', '4/3', '3/2', '14/9', '5/3', '16/9', '2/1', '24/10', '33/10', '18/5',  # Landscape
]

RATIOS = [eval(i) for i in LABELS]
RATIO2LABEL = dict(zip(RATIOS, LABELS))
LABEL2RATIO = dict(zip(LABELS, RATIOS))


def _num_pix(shape: ty.S[int]) -> int:
    """Return the number of elements in a 2D image."""
    assert len(shape) == 2
    return shape[0]*shape[1]


def _find_closest_multiple(i: ty.U[int, float], n: int = 32) -> int:
    """Return the closest multiple of `n` wrt the input `i`."""
    return round(i/n) * n


@torch.no_grad()
def aspect_ratio_aug(batch: ty.BatchData,
                     p: float = 1.0,
                     crop_min: float = 0.5,
                     crop_max: float = 1.0,
                     ref_shape: ty.N[ty.S[int]] = None) -> ty.BatchData:
    """Augmentation to change the aspect ratio of the input images.

    NOTE: Augmentation happens in-place!
    NOTE: If available, ground-truth depth maps are also resized. This is questionable and results in unreliable metrics.

    The augmentation consists of a crop augmentation followed by a resize. The crop augmentation extracts a centre
    crop from the images based on a sampled aspect ratio (see `RATIOS`). At least one of the dimensions (height or
    width) is guaranteed to be between [crop_min, crop_max] of the original image size. The resize augmentation adjusts
    the resolution of the extracted crop such that it has equal or fewer pixels than `ref_shape`.

    :param batch: (BatchData) Input training batch.
    :param p: (float) Probability of applying the augmentation [0, 1].
    :param crop_min: (float) Minimum relative size of the sampled crop [0, 1].
    :param crop_max: (float) Maximum relative size of the sampled crop [0, 1].
    :param ref_shape: (None|(int, int)) Reference shape to determine optimal resize. If `None` use original image shape.
    :return: (BatchData) Augmented training batch. Note that modification happens in-place.
    """
    sh = batch[0]['imgs'].shape[-2:]
    if random.random() > p:
        return resize_aug(batch, ref_shape, eps=1) if ref_shape and tuple(ref_shape) != tuple(sh) else batch
    ref_shape = ref_shape or sh
    batch = crop_aug(batch, min=crop_min, max=crop_max)
    batch = resize_aug(batch, ref_shape=ref_shape, eps=0.8)
    return batch


def crop_aug(batch: ty.BatchData, min: float = 0.5, max: float = 1.0) -> ty.BatchData:
    """Apply a centre crop with a random aspect ratio.

    :param batch: (BatchData) Input training batch.
    :param min: (float) Minimum relative size of the sampled crop [0, 1].
    :param max: (float) Maximum relative size of the sampled crop [0, 1].
    :return: (BatchData) Augmented training batch. Note that modification happens in-place.
    """
    x, y, m = batch
    shape = x['imgs'].shape[-2:]
    crop_shape, ratio = sample_crop(shape, min, max)
    fn = partial(KT.center_crop, size=crop_shape, mode='bilinear', align_corners=False)

    if 'augs' not in m: m['augs'] = []
    m['augs'].append(f'{list(shape)} -> {crop_shape} -> {RATIO2LABEL[ratio]}')

    # Speed up of 50% by processing all images simultaneously
    n, b = x['supp_imgs'].shape[:2]
    x['imgs'], y['imgs'], x['supp_imgs'], y['supp_imgs'] = fn(torch.cat((
        x['imgs'], y['imgs'],
        x['supp_imgs'].flatten(0, 1),
        y['supp_imgs'].flatten(0, 1),
    ))).split((b, b, n*b, n*b), dim=0)
    x['supp_imgs'] = x['supp_imgs'].unflatten(dim=0, sizes=(n, b))
    y['supp_imgs'] = y['supp_imgs'].unflatten(dim=0, sizes=(n, b))

    if 'depth' in y: y['depth'] = fn(y['depth'])
    if 'depth_hints' in y: y['depth_hints'] = fn(y['depth_hints'])

    if 'K' in y: y['K'] = geo.centre_crop_K(y['K'], crop_shape, shape)
    return x, y, m


def sample_crop(shape: ty.S[int], min: float = 0.5, max: float = 1.0) -> tuple[ty.S[int], float]:
    """Randomly sample a centre crop with a new aspect ratio.

    NOTE: In practice, we only guarantee that one of the dimensions will be between [min, max]. This is done to allow
    for additional flexibility when sampling aspect ratios that are very different from the original image.

    The general approach is to sample a random aspect ratio and coordinate for one of the dimensions (h or w). In order
    to ensure we have a uniform distribution of aspect ratios (independent of the original shape) we sample `n`
    possible crops with the same aspect ratio starting from both height and width. The final crop is randomly sampled
    from the resulting valid crops.

    :param shape: (int, int) Shape of the input image.
    :param min: (float) Minimum crop size [0, 1].
    :param max: (float) Maximum crop size [0, 1].
    :return: ((int, int), float) The sampled crop size and corresponding aspect ratio.
    """
    assert max >= min
    n = 10
    hs = torch.randint(int(shape[0]*min), int(shape[0]*max), (n,))
    ws = torch.randint(int(shape[1]*min), int(shape[1]*max), (n,))

    r = random.choice(RATIOS)
    hs, ws = torch.cat((hs, (ws/r).long())), torch.cat(((r*hs).long(), ws))  # Needs to be done simultaneously!

    valid = (hs >= 0) & (hs <= shape[0]) & (ws >= 0) & (ws <= shape[1])
    i = random.choice(valid.nonzero().squeeze())
    shape = hs[i].item(), ws[i].item()
    return shape, r


def resize_aug(batch: ty.BatchData, ref_shape: ty.S[int], eps: float = 0.8) -> ty.BatchData:
    """Apply a resize augmentation to match the number of pixels in `ref_shape`.

    NOTE: Resizing depth maps (especially sparse LiDAR) is questionable and will likely lead to unreliable metrics.

    :param batch: (BatchData) Input training batch.
    :param ref_shape: (int, int) Reference shape to match the number of pixels.
    :param eps: (float) Max percentage of ref_shape pixels to keep [0, 1].
    :return: (BatchData) Augmented training batch. Note that modification happens in-place.
    """
    x, y, m = batch
    new_shape = x['imgs'].shape[-2:]
    res_shape = sample_resize(new_shape, ref_shape, eps=eps)
    fn = partial(F.interpolate, size=res_shape, mode='bilinear', align_corners=False)

    if 'augs' not in m: m['augs'] = []
    m['augs'].append(str(res_shape))

    # Speed up of 50% by processing all images simultaneously
    n, b = x['supp_imgs'].shape[:2]
    x['imgs'], y['imgs'], x['supp_imgs'], y['supp_imgs'] = fn(torch.cat((
        x['imgs'], y['imgs'],
        x['supp_imgs'].flatten(0, 1),
        y['supp_imgs'].flatten(0, 1),
    ))).split((b, b, n*b, n*b), dim=0)
    x['supp_imgs'] = x['supp_imgs'].unflatten(dim=0, sizes=(n, b))
    y['supp_imgs'] = y['supp_imgs'].unflatten(dim=0, sizes=(n, b))

    # Resizing depth is questionable, especially if using sparse lidar...
    if 'depth' in y: y['depth'] = fn(y['depth'])
    if 'depth_hints' in y: raise RuntimeError(
        'Geometric augmentation should not be combined with depth hints... '
        'Interpolating depth is not well defined.'
    )

    if 'K' in y: y['K'] = geo.resize_K(y['K'], res_shape, shape=new_shape)
    return x, y, m


def sample_resize(shape: ty.S[int], ref_shape: ty.S[int], eps: float = 0.8) -> ty.S[int]:
    """Sample the resize shape for the new aspect ratio that provides the same number of pixels as `ref_shape`.

    NOTE: Sampled shape will always be a multiple of 32, as required by most networks. This also means the output shape
    will not exactly match the original aspect ratio, but it's close enough.

    :param shape: (int, int) Input image shape.
    :param ref_shape: (int, int) Reference shape to match number of pixels to.
    :param eps: (float) Max percentage of ref_shape pixels to keep [0, 1].
    :return: (int, int) Sampled resize shape for the input image.
    """
    mul = 32
    n, n_ref = _num_pix(shape), _num_pix(ref_shape)
    r = (n_ref/n) ** 0.5  # Scale factor to align number of pixels

    res_shape = [_find_closest_multiple(r*i, n=mul) for i in shape]
    while _num_pix(res_shape) > (n_ref*eps): res_shape = [i-mul for i in res_shape]
    return res_shape

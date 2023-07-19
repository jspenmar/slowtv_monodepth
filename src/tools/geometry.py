from functools import partial

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.feature import canny

import src.typing as ty
from . import ops

__all__ = [
    'extract_edges', 'to_scaled', 'to_log', 'to_inv', 'blend_stereo',
    'T_from_Rt', 'T_from_AAt',
    'BackprojectDepth', 'ProjectPoints', 'ViewSynth',
    'centre_crop_K', 'resize_K',
]


# DEPTH
# -----------------------------------------------------------------------------
def extract_edges(depth: ty.A,
                  preprocess: ty.N[str] = None,
                  sigma: int = 1,
                  mask: ty.N[ty.A] = None,
                  use_canny: bool = True) -> ty.A:
    """Detect edges in a dense LiDAR depth map.

    :param depth: (ndarray) (h, w, 1) Dense depth map to extract edges.
    :param preprocess: (str) Additional depth map post-processing. (log, inv, none)
    :param sigma: (int) Gaussian blurring sigma.
    :param mask: (None|ndarray) Optional boolean mask of valid pixels to keep.
    :param use_canny: (bool) If `True`, use `Canny` edge detection, otherwise `Sobel`.
    :return: (ndarray) (h, w) Detected depth edges in the image.
    """
    if preprocess not in {'log', 'inv', 'none', None}:
        raise ValueError(f'Invalid depth preprocessing. ({preprocess})')

    depth = depth.squeeze()
    if preprocess == 'log':
        depth = to_log(depth)
    elif preprocess == 'inv':
        depth = to_inv(depth)
        depth -= depth.min()  # Normalize disp to emphasize edges.
        depth /= depth.max()

    if use_canny:
        edges = canny(depth, sigma=sigma, mask=mask)
    else:
        depth = cv2.GaussianBlur(depth, (3, 3), sigmaX=sigma, sigmaY=sigma)
        dx = cv2.Sobel(src=depth, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
        dy = cv2.Sobel(src=depth, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)

        edges = np.sqrt(dx**2 + dy**2)[..., None]
        edges = edges > edges.mean()
        edges *= mask

    return edges


@ops.allow_np(permute=True)
def to_scaled(disp: ty.T, /, min: float = 0.01, max: ty.N[float] = 100) -> tuple[ty.T, ty.T]:
    """Convert a sigmoid depth map into a scaled depth & disparity.

    :param disp: (Tensor) (b, 1, h, w) Sigmoid disparity [0, 1].
    :param min: (float) Minimum depth value.
    :param max: (None|float) Maximum depth value.
    :return: (Tensor, Tensor) (b, 1, h, w) The scaled disparity/depth values.
    """
    if min <= 0: raise ValueError(f"Min depth must be greater than 0. ({min})")
    if max and (max < min): raise ValueError(f"Max depth must be greater than min. ({max} vs. {min})")
    i_max, i_min = 1/min, (1/max) if max else 0

    disp = (i_max - i_min)*disp + i_min
    return disp, to_inv(disp)


@ops.allow_np(permute=True)
def to_log(depth: ty.T, /) -> ty.T:
    """Convert linear depth into log depth."""
    depth = (depth > 0) * depth.clamp(min=ops.eps(depth)).log()
    return depth


@ops.allow_np(permute=True)
def to_inv(depth: ty.T, /) -> ty.T:
    """Convert linear depth into disparity."""
    disp = (depth > 0) / depth.clamp(min=ops.eps(depth))
    return disp


@ops.allow_np(permute=True)
def blend_stereo(disp_l: ty.T, disp_r: ty.T) -> ty.T:
    """Apply stereo disparity blending. Adapted from Monodepth
    (https://github.com/mrharicot/monodepth/blob/b76bee4bd12610b482163871b7ff93e931cb5331/monodepth_main.py#L56)

    As per the original Monodepth:
        - The 5% leftmost pixels are taken from the right disparity
        - The 5% rightmost pixels are taken from the left disparity
        - The remaining pixels are the average of both disparities

    :param disp_l: (Tensor) (*b, *1, h, w) The left disparities.
    :param disp_r: (Tensor) (*b, *1, h, w) The right (and flipped) disparities.
    :return: (Tensor) (*b, *1, h, w) The blended disparities.
    """
    s1, s2 = disp_l.shape, disp_r.shape
    n = disp_l.ndim
    if s1 != s2: raise ValueError(f'Non-matching shapes. ({s1} vs. {s2})')
    if n == 2: disp_l, disp_r = disp_l[None, None], disp_r[None, None]
    if n == 3: disp_l, disp_r = disp_l[None], disp_r[None]

    b, _, h, w = disp_l.shape

    x = torch.meshgrid(
        torch.linspace(0, 1, w, device=disp_l.device),
        torch.linspace(0, 1, h, device=disp_l.device),
        indexing='xy'
    )[0].expand(b, 1, -1, -1)

    mask_l = (20 * (x - 0.05)).clamp(min=0, max=1)  # 5% of the leftmost pixels (smooth)
    mask_r = mask_l.flip(dims=[-1])
    mask_mu = 1.0 - mask_l - mask_r

    disp_mu = (disp_l + disp_r) / 2
    disp = mask_r*disp_l + mask_l*disp_r + mask_mu*disp_mu
    if n == 2: disp = disp[0, 0]
    if n == 3: disp = disp[0]
    return disp
# -----------------------------------------------------------------------------


# TRANSFORMS
# -----------------------------------------------------------------------------
@ops.allow_np(permute=False)
def decompose_AA(aa: ty.T) -> tuple[ty.T, ty.T]:
    """Decompose axis-angle into each component. Angle: Magnitude, Axis: Direction."""
    angle = aa.norm(p=2, dim=-1, keepdim=True)  # (*, 1)
    axis = aa / angle.clip(min=ops.eps(angle))  # (*, 3)
    return axis, angle


@ops.allow_np(permute=False)
def R_from_q(q: ty.T) -> ty.T:
    """Compute rotation matrix from quaternion.

    :param q: (Tensor) (*, 4) Quaternion as (w, x, y, z).
    :return: (Tensor) (*, 3, 3) Rotation matrix.
    """
    s = q.shape
    if s[-1] != 4: raise ValueError(f"Incorrect `q` quaternion shape. ({s} vs. (*, 4))")

    w, x, y, z = torch.chunk(q, 4, dim=-1)
    R = torch.stack([
        torch.cat([1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w], dim=-1),
        torch.cat([2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w], dim=-1),
        torch.cat([2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2], dim=-1),
    ], dim=-2)
    return R


@ops.allow_np(permute=False)
def T_from_Rt(R: ty.T, t: ty.T) -> ty.T:
    """Construct transform matrix from Rotation and translation components.

    :param R: (Tensor) (*, 3, 3) Rotation matrix.
    :param t: (Tensor) (*, 3,) Translation vector.
    :return: (Tensor) (*, 4, 4) Transform matrix.
    """
    s1, s2 = R.shape, t.shape
    if R.ndim < 2 or s1[-2:] != (3, 3): raise ValueError(f"Incorrect `R` matrix shape. ({s1} vs. (*, 3, 3)")
    if s2[-1] != 3: raise ValueError(f"Incorrect `t` vector shape. ({s2} vs. (*, 3)")
    if s1[:-2] != s2[:-1]: raise ValueError(f"Non-matching shapes. ({s1} vs. {s2}")

    T = ops.eye_like(R.new_empty(s1[:-2]+(4, 4)))
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    return T


@ops.allow_np(permute=False)
def T_from_AAt(aa: ty.T, t: ty.T) -> ty.T:
    """Convert axis-angle rotation and translation vectors into a 4x4 transform matrix.
    Based on https://mathworld.wolfram.com/RodriguesRotationFormula.html

    :param aa: (Tensor) (*, 3) Axis-angle representation, where the norm represents the angle.
    :param t: (Tensor) (*, 3) Translation vector.
    :return: (Tensor) (*, 4, 4) Transform matrix.
    """
    s1, s2 = aa.shape, t.shape
    if s1[-1] != 3: raise ValueError(f"Incorrect `axisangle` shape. ({s1} vs. (*, 3)")
    if s2[-1] != 3: raise ValueError(f"Incorrect `t` shape. ({s2} vs. (*, 3)")
    if s1 != s2: raise ValueError(f"Non-matching shapes. ({s1} vs. {s2}")

    axis, angle = decompose_AA(aa)
    x, y, z = axis.chunk(3, dim=-1)  # [(*, 1)] * 3
    zr = torch.zeros_like(x)  # (*, 1)

    W = torch.stack([
        torch.cat([zr, -z, y, zr], dim=-1),
        torch.cat([z, zr, -x, zr], dim=-1),
        torch.cat([-y, x, zr, zr], dim=-1),
        torch.cat([zr, zr, zr, zr], dim=-1),
    ], dim=-2)  # (*, 4, 4)

    angle = angle.unsqueeze(-2)
    T = ops.eye_like(W) + W*angle.sin() + (W @ W)*(1 - angle.cos())
    T[..., :3, 3] = t
    return T


@ops.allow_np(permute=False)
def T_from_qt(q: ty.T, t: ty.T) -> ty.T:
    """Convert a quaternion rotation and translation vector into a 4x4 transform matrix.

    :param q: (Tensor) (b, 4) Quanternion representation as (w, x, y, z).
    :param t: (Tensor) (b, 3) Translation vector.
    :return: (Tensor) (b, 4, 4) Transform matrix.
    """
    s1, s2 = q.shape, t.shape
    if s1[-1] != 4: raise ValueError(f"Incorrect `axisangle` shape. ({s1} vs. (*, 3)")
    if s2[-1] != 3: raise ValueError(f"Incorrect `t` shape. ({s2} vs. (*, 3)")
    if s1[:-1] != s2[:-1]: raise ValueError(f"Non-matching shapes. ({s1} vs. {s2}")

    T = T_from_Rt(R_from_q(q), t)
    return T
# -----------------------------------------------------------------------------


# INTRINSICS
# -----------------------------------------------------------------------------
@ops.allow_np(permute=False)
def centre_crop_K(K: ty.T, new_shape: tuple[int, int], shape: ty.N[tuple[int, int]] = None) -> ty.T:
    """Adjust camera intrinsics when applying centre cropping.
    In practice, this just requires scaling the focal length/principal point based on the ratio of the crop.

    :param K: (Tensor) (*, 4, 4) Camera intrinsic parameters.
    :param new_shape: (int, int) New image shape as `(h, w)`.
    :param shape: (None|(int, int) Old image shape as `(h, w)`. If `None`, assume `K` is normalized.
    :return: (Tensor) (*, 4, 4) Intrinsics with adjusted focal lengths.
    """
    if shape is None: shape = (1, 1)
    K = K.clone()
    K[..., 0, 2] *= new_shape[1]/shape[1]
    K[..., 1, 2] *= new_shape[0]/shape[0]
    return K


@ops.allow_np(permute=False)
def resize_K(K: ty.T, new_shape: tuple[int, int], shape: ty.N[tuple[int, int]] = None) -> ty.T:
    """Adjust camera intrinsics when resizing the image.
    In practice, this just requires scaling the focal length/principal point based on the ratio of the crop.

    :param K: (Tensor) (*, 4, 4) Camera intrinsic parameters.
    :param new_shape: (int, int) New image shape as `(h, w)`.
    :param shape: (None|(int, int) Old image shape as `(h, w)`. If `None`, assume `K` is normalized.
    :return: (Tensor) (*, 4, 4) Intrinsics with adjusted focal lengths.
    """
    if shape is None: shape = (1, 1)
    K = K.clone()
    K[..., 0, :] *= new_shape[1] / shape[1]
    K[..., 1, :] *= new_shape[0] / shape[0]
    return K


@ops.allow_np(permute=False)
def pad_K(K: ty.T) -> ty.T:
    """Pad K into a 4x4 matrix.

    :param K: (Tensor) (*, 3|4, 3|4) Camera intrinsic parameters.
    :return: (Tensor) (*, 4, 4) Padded intrinsics.
    """
    *sh, h, w = K.shape[-2:]
    if h not in {3, 4} and w not in {3, 4}: raise ValueError(f"Incorrect shape. ({K.shape} vs. (*, 3|4, 3|4)).")

    if h == 3: K = torch.cat((K, K.new_zeros(*sh, 1, w)), dim=-2)
    if w == 3: K = torch.cat((K, K.new_zeros(*sh, 4, 1)), dim=-1)  # We are guaranteed that `h = 4`.
    K[..., 3, 3] = 1
    return K
# -----------------------------------------------------------------------------


# REPROJECTION
# -----------------------------------------------------------------------------
class BackprojectDepth(nn.Module):
    """Module to backproject a depth map into a pointcloud.
    This is defined as: D K^{-1} [x y 1]

    :param shape: (int, int) Depth map shape as (height, width).
    """
    def __init__(self, shape: tuple[int, int]):
        super().__init__()
        self.h, self.w = shape
        self.register_buffer('ones', torch.ones(1, 1, self.h*self.w), persistent=False)
        self.register_buffer('pix', self._get_pix_grid(), persistent=False)

    def _get_pix_grid(self) -> ty.T:
        """Precompute homogenous meshgrid as (1, 3, h*w) as (x, y, 1)."""
        grid = torch.meshgrid(torch.arange(self.w), torch.arange(self.h), indexing='xy')  # (h, w), (h, w)
        pix = torch.stack(grid).view(2, -1)[None]  # (1, 2, h*w) as (x, y)
        pix = torch.cat((pix, self.ones), dim=1)  # (1, 3, h*w)
        return pix

    def forward(self, depth: ty.T, K_inv: ty.T) -> ty.T:
        """Backproject a depth map into a pointcloud.
        Camera is assumed to be at the origin.

        :param depth: (Tensor) (b, 1, h, w) Depth map to backproject.
        :param K_inv: (Tensor) (b, 4, 4) Inverse camera intrinsic parameters.
        :return: (Tensor) (b, 4, h*w) Backprojected 3-D points as (x, y, z, 1).
        """
        b = depth.shape[0]
        pts = K_inv[:, :3, :3] @ self.pix.repeat(b, 1, 1)  # (b, 3, h*w) Cam rays.
        pts *= depth.flatten(-2)  # 3D points.
        pts = torch.cat((pts, self.ones.repeat(b, 1, 1)), dim=1)  # (b, 4, h*w) Add homogenous.
        return pts


class ProjectPoints(nn.Module):
    """Convert a 3-D pointcloud into image grid sample locations.
    This is defined as: K T [x y z 1]

    :param shape: (int, int) Depth map shape as (height, width).
    """
    def __init__(self, shape: tuple[int, int]):
        super().__init__()
        self.h, self.w = shape

    def forward(self, pts: ty.T, K: ty.T) -> tuple[ty.T, ty.T]:
        """Convert a 3-D pointcloud into image grid sample locations.

        :param pts: (Tensor) (b, 4, h*w) Pointcloud points to project.
        :param K:  (Tensor) (b, 4, 4) Camera intrinsic parameters.
        :return: (
            pix_coords: (Tensor) (b, h, w, 2) Grid sample locations [-1, 1] as (x, y).
            cam_depth: (Tensor) (b, 1, h, w) Depth map in the transformed reference frame.
        )
        """
        pts = pts[:, :3]  # Remove homo.
        depth = pts[:, 2:].clamp(min=ops.eps(pts))
        pix = (K[:, :3, :3] @ (pts/depth.clamp(min=0.1)))[:, :2]  # Project & remove homo.

        # Reshape as image.
        depth = depth.view(-1, 1, self.h, self.w)
        grid = pix.view(-1, 2, self.h, self.w).permute(0, 2, 3, 1)  # (b, h, w, 2) as (x, y)

        grid[..., 0] /= self.w-1  # Normalize [0, 1].
        grid[..., 1] /= self.h-1
        grid = (grid - 0.5) * 2  # Normalize [-1, 1].
        return grid, depth


class ViewSynth(nn.Module):
    """Warp an image according to depth and pose information.
    Combines the process of backprojecting into a pointcloud and reprojecting onto the new image.

    :param shape: (int, int) Depth map shape as (h, w).
    """
    def __init__(self, shape: tuple[int, int]):
        super().__init__()
        self.shape = shape
        self.backproj = BackprojectDepth(shape)
        self.proj = ProjectPoints(shape)
        self.sample = partial(F.grid_sample, mode='bilinear', padding_mode='border', align_corners=False)

    def forward(self, input: ty.T, depth: ty.T, T: ty.T, K: ty.T, K_inv: ty.N[ty.T] = None) -> tuple[ty.T, ty.T, ty.T]:
        """Warp the input image according to a source depth map.

        NOTE: In practice, this means we are warping the SUPPORT FRAMES, such that their appearance matches the TARGET.
        `T` is used to transform points from the SOURCE reference frame to the TARGET reference frame.

        :param input: (Tensor) (b, *, h, w) Input tensor to warp.
        :param depth: (Tensor) (b, 1, h, w) Predicted depth for the source.
        :param T: (Tensor) (b, 4, 4) Pose of the target image relative to the source.
        :param K: (Tensor) (b, 4, 4) The target image camera intrinsics.
        :param K_inv: (None|Tensor) (b, 4, 4) The source image inverse intrinsics. If `None` use `K.inverse()`.
        :return: (
            input_warp: (Tensor) (b, *, h, w) The synthesized warped image.
            depth_warp: (Tensor) (b, 1, h, w) The synthesized warped depth map (in the TARGET reference frame).
            mask_valid: (Tensor) (b, 1, h, w) Boolean mask indicating valid warp pixels.
        )
        """
        if K_inv is None: K_inv = K.inverse()

        pts = self.backproj(depth, K_inv)
        pts = T @ pts  # Transform to new reference frame.
        grid, depth_warp = self.proj(pts, K)
        mask_valid = (grid.abs() < 1).all(dim=-1, keepdim=True).permute(0, 3, 1, 2)
        input_warp = self.sample(input=input, grid=grid)

        return input_warp, depth_warp, mask_valid
# -----------------------------------------------------------------------------

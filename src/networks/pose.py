"""Relative pose prediction network."""
import timm
import torch
import torch.nn as nn

import src.typing as ty
from src import register
from src.tools import ops

__all__ = ['PoseNet']


@register('pose')
class PoseNet(nn.Module):
    """Relative pose prediction network. From SfM-Learner (https://arxiv.org/abs/1704.07813).

    The objective is to predict the relative pose between two images. The network consists of an encoder
    (with duplicated/scaled weights at the input layer) and a light regression decoder. Pose is predicted
    as an axis-angle rotation (direction=axis, magnitude=angle) and a translation vector.

    NOTE: Translation is not in metric scale unless training with stereo+mono or velocity information.

    Optionally, we can also predict camera intrinsics (focal length and principal point) using two separate decoders.
    Based on DiW (https://arxiv.org/abs/1904.04998). These intrinsics are predicted as normalized and must be scaled
    using the input image size. The focal length decoder uses a `softplus` activation to ensure a positive output,
    while the principal point uses `sigmoid` to ensure it lies within the image (this isn't strictly required,
    but may make optimization easier).

    :param enc_name: (str) `timm` encoder key (check `timm.list_models()`).
    :param learn_K: (bool) If `True`, add decoders to predict camera focal length and principal point.
    :param pretrained: (bool) If `True`, load an encoder pretrained on ImageNet.
    """
    def __init__(self, enc_name: str = 'resnet18', learn_K: bool = False, pretrained: bool = False):
        super().__init__()
        self.enc_name = enc_name
        self.learn_K = learn_K
        self.pretrained = pretrained

        self.n_imgs = 2
        self.encoder = timm.create_model(enc_name, in_chans=3 * self.n_imgs, features_only=True, pretrained=pretrained)
        self.n_ch_enc = self.encoder.feature_info.channels()
        self.n_ch_dec = 256

        self.pose_eps = 0.01

        self.squeeze = self.block(self.n_ch_enc[-1], self.n_ch_dec, kernel_size=1)
        self.decoders = nn.ModuleDict({'pose': self._get_pose_dec(self.n_ch_dec, self.n_imgs)})
        if self.learn_K:
            self.decoders['focal'] = self._get_focal_dec(self.n_ch_dec)
            self.decoders['offset'] = self._get_offset_dec(self.n_ch_dec)

    @staticmethod
    def block(in_ch: int, out_ch: int, kernel_size: int, stride: int = 1, padding: int = 0) -> nn.Module:
        """Return a Conv+ReLU block."""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding),
            nn.ReLU(inplace=True)
        )

    @staticmethod
    def build_K(fs: ty.T, cs: ty.T) -> ty.T:
        """Construct camera intrinsics matrix from focal length and principal point.

        :param fs: (Tensor) (b, 2) Focal length as (x, y).
        :param cs: (Tensor) (b, 2) Principal point as (x, y).
        :return: (Tensor) (b, 4, 4) Camera intrinsics matrix.
        """
        K = ops.expand_dim(torch.eye(4, device=fs.device), num=fs.shape[0], dim=0, insert=True).clone()
        K[:, 0, 0] = fs[:, 0]
        K[:, 1, 1] = fs[:, 1]
        K[:, 0, 2] = cs[:, 0]
        K[:, 1, 2] = cs[:, 1]
        return K

    def _get_pose_dec(self, n_ch: int, n_imgs: int) -> nn.Sequential:
        """Return the pose estimation decoder. (b, c, h, w) -> (b, n_imgs, 6)"""
        return nn.Sequential(
            self.block(n_ch, n_ch, kernel_size=3, stride=1, padding=1),
            self.block(n_ch, n_ch, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(n_ch, 6*n_imgs, kernel_size=1),
            nn.AdaptiveAvgPool2d((1, 1)),  # (b, 6*n, 1, 1)
            nn.Flatten(),  # (b, 6*n)
            nn.Unflatten(dim=-1, unflattened_size=(n_imgs, 6)),  # (b, n, 6)
        )

    def _get_focal_dec(self, n_ch: int) -> nn.Sequential:
        """Return focal length estimation decoder. (b, c, h, w) -> (b, 2)"""
        return nn.Sequential(
            self.block(n_ch, n_ch, kernel_size=3, stride=1, padding=1),
            self.block(n_ch, n_ch, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(n_ch, 2, kernel_size=1),  # (b, 2, h, w)
            nn.AdaptiveAvgPool2d((1, 1)),  # (b, 2, 1, 1)
            nn.Flatten(),  # (b, n)
            nn.Softplus(),
        )

    def _get_offset_dec(self, n_ch: int) -> nn.Sequential:
        """Return principal point estimation decoder. (b, c, h, w) -> (b, 2)"""
        return nn.Sequential(
            self.block(n_ch, n_ch, kernel_size=3, stride=1, padding=1),
            self.block(n_ch, n_ch, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(n_ch, 2, kernel_size=1),  # (b, 2, h, w)
            nn.AdaptiveAvgPool2d((1, 1)),  # (b, 2, 1, 1)
            nn.Flatten(),  # (b, n)
            nn.Sigmoid(),
        )

    def forward(self, x: ty.T) -> ty.PosePred:
        """Run pose estimation forward poss.

        NOTE: The `n_imgs` dimension in the pose prediction is a legacy from the original SfM-Learner and Monodepth2.
        It should have been removed when re-implementing the benchmark, but removing it now would break all the
        checkpoints. This would also require some refactoring in the `Trainer`.

        :param x: (Tensor) (b, n_imgs*3, h, w) Channel-wise concatenated input images (n_imgs=2).
        :return: {
            R: (b, n_imgs, 3) Predicted rotation as axis-angle (direction=axis, magnitude=angle).
            t: (b, n_imgs, 3) Predicted translation.

            (Optional) (If using `learn_K`)
            fs: (b, 2) Predicted normalized focal lengths as (x, y).
            cs: (b, 2) Predicted normalized principal point as (x, y).
        }
        """
        feat = self.encoder(x)
        feat = self.squeeze(feat[-1])

        out = self.pose_eps * self.decoders['pose'](feat)
        out = {'R': out[..., :3], 't': out[..., 3:]}

        if self.learn_K:
            out['fs'] = self.decoders['focal'](feat)
            out['cs'] = self.decoders['offset'](feat)

        return out

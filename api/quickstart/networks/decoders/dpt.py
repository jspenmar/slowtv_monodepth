"""Adapted from MiDaS (https://github.com/isl-org/MiDaS/tree/master/midas)"""
from typing import Optional

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .utils import ACT

__all__ = ['DptDecoder']


class ResidualBlock(nn.Module):
    """Residual convolution module."""
    def __init__(self, ch: int, act: nn.Module, use_bn: bool = False):
        super().__init__()
        self.bn = use_bn

        self.conv1 = nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=True, groups=1)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=True, groups=1)

        if self.bn:
            self.bn1 = nn.BatchNorm2d(ch)
            self.bn2 = nn.BatchNorm2d(ch)

        self.act = act
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x: Tensor) -> Tensor:
        out = self.act(x)
        out = self.conv1(out)
        if self.bn: out = self.bn1(out)

        out = self.act(out)
        out = self.conv2(out)
        if self.bn: out = self.bn2(out)

        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    """Feature fusion block."""
    def __init__(self,
                 ch: int,
                 act: nn.Module,
                 deconv: bool = False,
                 use_bn: bool = False,
                 expand: bool = False,
                 align_corners: bool = True,
                 size: Optional[tuple[int, int]] = None):
        super().__init__()
        self.deconv = deconv
        self.align_corners = align_corners
        self.expand = expand
        self.size = size

        out_ch = ch
        if self.expand: out_ch //= 2

        self.resConfUnit1 = ResidualBlock(ch, act, use_bn)
        self.resConfUnit2 = ResidualBlock(ch, act, use_bn)
        self.out_conv = nn.Conv2d(ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True, groups=1)

        self.skip_add = nn.quantized.FloatFunctional()

    def upsample(self, x: Tensor, size: Optional[tuple[int, int]] = None) -> Tensor:
        if size: kw = {'size': size}
        elif self.size: kw = {'size': self.size}
        else: kw = {'scale_factor': 2}

        return F.interpolate(x, **kw, mode='bilinear', align_corners=self.align_corners)

    def forward(self, *xs: Tensor, size: Optional[tuple[int, int]] = None) -> Tensor:
        out = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            out = self.skip_add.add(out, res)

        out = self.resConfUnit2(out)
        out = self.upsample(out, size=size)
        out = self.out_conv(out)
        return out


class DptDecoder(nn.Module):
    def __init__(self,
                 num_ch_enc: list[int],
                 enc_sc: list[int],
                 upsample_mode: str = 'nearest',
                 use_skip: bool = True,
                 out_sc: list[int] = (0, 1, 2, 3),
                 out_ch: int = 1,
                 out_act: str = 'relu'):
        super().__init__()
        self.num_ch_enc = num_ch_enc
        self.enc_sc = enc_sc
        self.upsample_mode = upsample_mode
        self.use_skip = use_skip
        self.out_sc = out_sc
        self.out_ch = out_ch
        self.out_act = out_act

        self.activation = ACT[self.out_act]
        self.num_ch_dec = 256

        self.layers = nn.ModuleList([
            nn.Conv2d(ch, self.num_ch_dec, kernel_size=3, stride=1, padding=1, bias=False) for ch in self.num_ch_enc
        ])

        self.refine = nn.ModuleList([
            self._make_fusion_block(self.num_ch_dec, use_bn=False) for _ in self.num_ch_enc
        ])

        self.out_conv = nn.ModuleDict(
            {str(i): self._make_head(self.num_ch_dec, self.out_ch, self.activation, hidden_ch=32) for i in self.out_sc}
        )

    @staticmethod
    def _make_fusion_block(ch: int, use_bn: bool, size: Optional[tuple[int, int]] = None):
        return FeatureFusionBlock(
            ch,
            act=nn.ReLU(False),
            deconv=False,
            use_bn=use_bn,
            expand=False,
            align_corners=True,
            size=size,
        )

    @staticmethod
    def _make_head(in_ch: int, out_ch: int, act: nn.Module, hidden_ch: Optional[int] = None) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 2, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_ch // 2, hidden_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(hidden_ch, out_ch, kernel_size=1, stride=1, padding=0),
            act,
        )

    def forward(self, feat: list[Tensor]) -> dict[int, Tensor]:
        feat = [conv(f) for conv, f in zip(self.layers, feat)]
        out = {}

        x = feat[-1]
        n = len(feat) - 1
        for i in range(n, -1, -1):
            xs = [x] if i == n else [x, feat[i]]
            x = self.refine[i](*xs)

            if i in self.out_sc: out[i] = self.out_conv[str(i)](x)

        return out

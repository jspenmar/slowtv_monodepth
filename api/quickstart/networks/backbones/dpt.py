"""Tools to create DPT encoders from `timm` backbones.
Adapted from MiDaS (https://github.com/isl-org/MiDaS/tree/master/midas/backbones)
"""
import torch
import torch.nn as nn

import src.typing as ty
from . import (
    forward_beit, forward_swin, forward_vit,
    make_beitl16_512, make_beitl16_384, make_beitb16_384,
    make_swinl12_384, make_swin2l24_384, make_swin2b24_384, make_swin2t16_256,
    make_vitb_rn50_384, make_vitl16_384, make_vitb16_384,
)

__all__ = ['DptEncoder']


class FeatureInfo:
    """Encoder multi-scale feature information. Used for compatibility with `timm`."""
    def __init__(self, n_ch: ty.S[int]):
        self.n_ch = n_ch
        self.red = [32//(2**i) for i in range(len(self.n_ch)-1, -1, -1)]  # Final stage is downsampled by 32

    def channels(self) -> ty.S[int]: return self.n_ch
    def reduction(self) -> ty.S[int]: return self.red


class DptEncoder(nn.Module):
    def __init__(self, enc_name: str, pretrained: bool = True, use_readout: str = 'project'):
        super().__init__()

        n, pt, r = enc_name, pretrained, use_readout
        if n == 'beitl16_512': self.net = make_beitl16_512(pt, hooks=[5, 11, 17, 23], use_readout=r)
        elif n == 'beitl16_384': self.net = make_beitl16_384(pt, hooks=[5, 11, 17, 23], use_readout=r)
        elif n == 'beitb16_384': self.net = make_beitb16_384(pt, hooks=[2, 5, 8, 11], use_readout=r)

        elif n == 'swin2l24_384': self.net = make_swin2l24_384(pt, hooks=[1, 1, 17, 1])  # [0,1], [0,1], [0,17], [0,1]
        elif n == 'swin2b24_384': self.net = make_swin2b24_384(pt, hooks=[1, 1, 17, 1])  # [0,1], [0,1], [0,17], [0,1]
        elif n == 'swin2t16_256': self.net = make_swin2t16_256(pt, hooks=[1, 1, 5, 1])   # [0,1], [0,1], [0, 5], [0,1]

        elif n == 'swinl12_384': self.net = make_swinl12_384(pt, hooks=[1, 1, 17, 1])    # [0,1], [0,1], [0,17], [0,1]

        elif n == 'vitb_rn50_384': self.net = make_vitb_rn50_384(pt, hooks=[0, 1, 8, 11], use_readout=r)
        elif n == 'vitl16_384': self.net = make_vitl16_384(pt, hooks=[5, 11, 17, 23], use_readout=r)
        elif n == 'vitb16_384': self.net = make_vitb16_384(pt, hooks=[2, 5, 8, 11], use_readout=r)
        else: raise ValueError(f"Backbone '{n}' not implemented")

        self.feature_info = FeatureInfo(self.net.n_ch)

        if 'beit' in n: self.fwd = forward_beit
        elif 'swin' in n: self.fwd = forward_swin
        elif 'vit' in n: self.fwd = forward_vit

    def forward(self, x: ty.T) -> ty.S[ty.T]:
        return self.fwd(self.net, x.to(memory_format=torch.channels_last))

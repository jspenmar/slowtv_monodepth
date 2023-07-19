from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

ACT = {
    'sigmoid': nn.Sigmoid(),
    'relu': nn.ReLU(inplace=True),
    'none': nn.Identity(),
    None: nn.Identity(),
}


def _is_contiguous(tensor: Tensor) -> bool:
    if torch.jit.is_scripting(): return tensor.is_contiguous()
    else: return tensor.is_contiguous(memory_format=torch.contiguous_format)


class LayerNorm2d(nn.LayerNorm):
    """From timm."""
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__(normalized_shape, eps=eps)

    def forward(self, x) -> Tensor:
        if _is_contiguous(x):
            return F.layer_norm(
                x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps
            ).permute(0, 3, 1, 2)
        else:
            s, u = torch.var_mean(x, dim=1, keepdim=True)
            x = (x-u) * torch.rsqrt(s + self.eps)
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
            return x


def conv1x1(in_ch: int, out_ch: int, bias: bool = True) -> nn.Conv2d:
    """Conv layer with 1x1 kernel."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=(1, 1), bias=bias)


def conv3x3(in_ch: int, out_ch: int, bias: bool = True) -> nn.Conv2d:
    """Conv layer with 3x3 kernel and `reflect` padding."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=(3, 3), padding=1, padding_mode='reflect', bias=bias)


def conv_block(in_ch: int, out_ch: int) -> nn.Module:
    """Layer to perform a convolution followed by ELU."""
    return nn.Sequential(OrderedDict({
        'conv': conv3x3(in_ch, out_ch),
        'act': nn.ELU(inplace=True),
    }))

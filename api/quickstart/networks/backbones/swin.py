"""Adapted from MiDaS (https://github.com/isl-org/MiDaS/tree/master/midas/backbones)"""
import numpy as np
import timm
import torch
import torch.nn as nn

from .utils import Transpose, activations, forward_default, get_activation

__all__ = ['forward_swin', 'make_swinl12_384', 'make_swin2l24_384', 'make_swin2b24_384', 'make_swin2t16_256']


def forward_swin(net, x):
    return forward_default(net, x)


def make_swinl12_384(pretrained, hooks=(1, 1, 17, 1)):
    model = timm.create_model('swin_large_patch4_window12_384', pretrained=pretrained)
    return _make_swin_backbone(model, hooks=hooks)


def make_swin2l24_384(pretrained, hooks=(1, 1, 17, 1)):
    model = timm.create_model('swinv2_large_window12to24_192to384_22kft1k', pretrained=pretrained)
    return _make_swin_backbone(model, hooks=hooks)


def make_swin2b24_384(pretrained, hooks=(1, 1, 17, 1)):
    model = timm.create_model('swinv2_base_window12to24_192to384_22kft1k', pretrained=pretrained)
    return _make_swin_backbone(model, hooks=hooks)


def make_swin2t16_256(pretrained, hooks=(1, 1, 17, 1)):
    model = timm.create_model('swinv2_tiny_window16_256', pretrained=pretrained)
    return _make_swin_backbone(model, hooks=hooks, patch_grid=[64, 64])


def _make_swin_backbone(model, hooks=(1, 1, 17, 1), patch_grid=(96, 96)):
    net = nn.Module()

    net.model = model
    net.model.layers[0].blocks[hooks[0]].register_forward_hook(get_activation('1'))
    net.model.layers[1].blocks[hooks[1]].register_forward_hook(get_activation('2'))
    net.model.layers[2].blocks[hooks[2]].register_forward_hook(get_activation('3'))
    net.model.layers[3].blocks[hooks[3]].register_forward_hook(get_activation('4'))
    net.activations = activations

    patch_grid = np.array(getattr(model, 'patch_grid', patch_grid), dtype=int)
    net.act_postprocess1 = nn.Sequential(Transpose(1, 2), nn.Unflatten(2, torch.Size(patch_grid)))
    net.act_postprocess2 = nn.Sequential(Transpose(1, 2), nn.Unflatten(2, torch.Size(patch_grid//2)))
    net.act_postprocess3 = nn.Sequential(Transpose(1, 2), nn.Unflatten(2, torch.Size(patch_grid//4)))
    net.act_postprocess4 = nn.Sequential(Transpose(1, 2), nn.Unflatten(2, torch.Size(patch_grid//8)))
    return net

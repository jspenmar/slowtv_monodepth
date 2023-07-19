"""Monocular depth estimation network."""
from typing import Optional, Union, Type

import timm
import torch.nn as nn
from torch import Tensor

from .backbones import DptEncoder
from. decoders import DECODERS

__all__ = ['DepthNet']

MASKS = {None: None, 'explainability': 'sigmoid', 'uncertainty': 'relu'}
VALID_MASKS = set(MASKS)


class DepthNet(nn.Module):
    """Monocular depth estimation network.

    The objective is to predict depth from a single image. Technically we predict disparity (i.e. inverse depth) and
    clip it to the range [0, 1]. The model can provide intermediate multi-scale outputs used as auxiliary supervision.
    We optionally provide masking prediction, typically used to weight the photometric reconstruction loss.

    Features:
        - Virtual Stereo: From Monodepth (https://arxiv.org/abs/1609.03677). Predicts a disparity map for the
            stereo pair, using only the input monocular image. This is used to compute additional consistency losses.
            Unlike the original implementation, we do not assume the input is always the left image. As such, we
            always predict both a left and right virtual disparity, which should be indexed accordingly in the Trainer.

        - Virtual Stereo Blending: From SuperDepth (https://arxiv.org/abs/1810.01849). Computes the predictions for the
        horizontally flipped image and blends them with the original predictions. This is the same post-processing from
        Monodepth, but applied during training.

        - Masking: Predicts a mask used to estimate the uncertainty of the depth prediction or to weight the photometric
            reconstruction loss. Incorporates various types:
                + Explainability: From SfMLearner (https://arxiv.org/abs/1704.07813). Predicted reliability for the
                    photometric loss at each pixel. A LOW value represents low confidence, meaning that it is removed
                    from the loss. Requires an additional binary cross-entropy regularization pushing all values to 1.

                + Uncertainty: From Klodt (https://arxiv.org/abs/1904.04998). Predicts the uncertainty for each pixel.
                    A HIGH value represents low confidence, meaning that it is removed from the loss. Requires an
                    additional regularization (sum of uncertainties), computed in the photometric loss.

    NOTE: Masking and intrinsics estimation is slightly different from the original papers, where they are predicted
    by the pose network. We follow Monodepth2 and predict them using the depth network.

    :param enc_name: (str) `timm` or `dpt` encoder key (check `timm.list_models()`).
    :param pretrained: (bool) If `True`, load an encoder pretrained on ImageNet.
    :param dec_name: (str) Custom decoder type to use.
    :param out_scales: (int|list[int]) Multi-scale outputs to return as `2**s`.
    :param use_virtual_stereo: (bool) If `True`, output a disparity prediction for the stereo image.
    :param mask_name: (None|str) Optional mask to predict to weight the reconstruction loss. {None, 'explainability', 'uncertainty'}.
    :param num_ch_mask: (None|int) Number of `supp_imgs` to predict masks for.
    :param use_stereo_blend: (bool) If `True`, run a forward pass on the horizontally flipped images and blend.
    """
    def __init__(self,
                 enc_name: str = 'resnet18',
                 pretrained: bool = True,
                 dec_name: str = 'monodepth',
                 out_scales: Union[int, list[int]] = (0, 1, 2, 3),
                 mask_name: Optional[str] = None,
                 num_ch_mask: Optional[int] = None,
                 use_virtual_stereo: bool = False,
                 use_stereo_blend: bool = False):
        super().__init__()
        self.enc_name = enc_name
        self.pretrained = pretrained
        self.dec_name = dec_name
        self.out_scales = [out_scales] if isinstance(out_scales, int) else out_scales
        self.mask_name = mask_name
        self.num_ch_mask = num_ch_mask
        self.use_virtual_stereo = use_virtual_stereo
        self.use_stereo_blend = use_stereo_blend

        if self.dec_name not in DECODERS:
            raise KeyError(f'Invalid decoder. ({self.dec_name} vs. {list(DECODERS)}')

        if self.mask_name not in VALID_MASKS:
            raise KeyError(f'Invalid mask. ({self.mask_name} vs. {VALID_MASKS}')

        if self.dec_name == 'ddvnet' and self.mask_name is not None:
            raise KeyError(f'DDVNet is not compatible with mask prediction.')

        if self.mask_name and self.num_ch_mask <= 0:
            raise ValueError(f'Invalid number of mask channels. ({self.num_ch_mask} vs. >=1)')

        self.encoder, self.num_ch_enc, self.enc_sc = self._get_encoder()

        cls = DECODERS[self.dec_name]
        self.decoders = nn.ModuleDict({
            'disp': self._get_depth_decoder(cls),
            **({f'mask': self._get_mask_decoder(cls)} if self.mask_name else {}),
        })

    def _get_encoder(self) -> tuple[nn.Module, list[int], list[int]]:
        """Return the timm/DPT encoder and feature information. (b, 3, h, w) -> [(b, c, h/2**s, w/2**s)]"""
        net = DptEncoder(self.enc_name.replace('dpt_', ''), self.pretrained) if self.enc_name.startswith('dpt_') \
            else timm.create_model(self.enc_name, features_only=True, pretrained=self.pretrained)

        return net, net.feature_info.channels(), net.feature_info.reduction()

    def _get_depth_decoder(self, cls: Type[nn.Module]) -> nn.Module:
        """Return the depth prediction decoder. (b, c, h, w) -> {s: (b, 1, h/2**s, w/2**s)}"""
        return cls(
            num_ch_enc=self.num_ch_enc, enc_sc=self.enc_sc,
            upsample_mode='nearest', use_skip=True,
            out_sc=self.out_scales, out_ch=1 + (2*self.use_virtual_stereo), out_act='sigmoid'
        )

    def _get_mask_decoder(self, cls: Type[nn.Module]) -> nn.Module:
        """Return the mask prediction decoder. (b, c, h, w) -> {s: (b, n, h/2**s, w/2**s)}"""
        return cls(
            num_ch_enc=self.num_ch_enc, enc_sc=self.enc_sc,
            upsample_mode='nearest', use_skip=True,
            out_sc=self.out_scales, out_ch=self.num_ch_mask, out_act=MASKS[self.mask_name]
        )

    def forward(self, x: Tensor):
        """Run depth estimation forward poss (with optional blending).

        :param x: (Tensor) (b, 3, h, w) Input image.
        :return: {
            depth_feats: (list[Tensor]) (b, c, h/2**s, w/2**s) Multi-scale depth encoder features.
            disp: (TensorDict) (b, 1, h/2**s, w/2**s) Multi-scale sigmoid disparity predictions.

            (Optional)
            (If using `use_virtual_stereo`)
            disp_stereo: (TensorDict) (b, 2, h/2**s, w/2**s) Multi-scale virtual stereo sigmoid disparity predictions.

            (If using `mask_name`)
            mask: (TensorDict) (b, n, h/2**s, w/2**s) Multi-scale mask predictions.

            (If using `use_virtual_stereo` and `mask_name`)
            mask_stereo: (TensorDict) (b, n, h/2**s, w/2**s) Multi-scale virtual stereo mask predictions.
        }
        """
        out = {}
        out['depth_feats'] = feat = self.encoder(x)

        for k, dec in self.decoders.items():
            o = dec(feat)
            out[k] = {k: o[k] for k in sorted(o)}

        if self.use_virtual_stereo:
            out['disp_stereo'] = {k2: v2[:, 1:] for k2, v2 in out['disp'].items()}
            out['disp'] = {k2: v2[:, :1] for k2, v2 in out['disp'].items()}

        return out

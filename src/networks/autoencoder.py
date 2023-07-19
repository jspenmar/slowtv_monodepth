"""Image autoencoder network."""
import timm
import torch.nn as nn

import src.typing as ty
from src import register, DEC_REG
from src.utils import sort_dict

__all__ = ['AutoencoderNet']


@register('autoencoder')
class AutoencoderNet(nn.Module):
    """Image autoencoder network. From FeatDepth (https://arxiv.org/abs/2007.10603).

    Heavily based on the Depth network with some changes:
        - Does not accept DPT encoders
        - Single decoder
        - Produces 3 sigmoid channels (RGB)
        - No skip connections, it's an autoencoder!

    :param enc_name: (str) `timm` encoder key (check `timm.list_models()`).
    :param pretrained: (bool) If `True`, load an encoder pretrained on ImageNet.
    :param dec_name: (str) Custom decoder class to use.
    :param out_scales: (int|list[int]) Multi-scale outputs to return as `2**s`.
    """
    def __init__(self,
                 enc_name: str = 'resnet18',
                 pretrained: bool = True,
                 dec_name: str = 'monodepth',
                 out_scales: ty.U[int, ty.S[int]] = (0, 1, 2, 3)):
        super().__init__()
        self.enc_name = enc_name
        self.pretrained = pretrained
        self.dec_name = dec_name
        self.out_scales = [out_scales] if isinstance(out_scales, int) else out_scales

        if self.dec_name not in DEC_REG:
            raise KeyError(f'Invalid decoder key. ({self.dec_name} vs. {DEC_REG.keys()}')

        self.encoder = timm.create_model(self.enc_name, features_only=True, pretrained=pretrained)
        self.num_ch_enc = self.encoder.feature_info.channels()
        self.enc_sc = self.encoder.feature_info.reduction()

        self.decoder = DEC_REG[self.dec_name](
            num_ch_enc=self.num_ch_enc, enc_sc=self.enc_sc,
            upsample_mode='nearest', use_skip=False,
            out_sc=self.out_scales, out_ch=3, out_act='sigmoid'
        )

    def forward(self, x: ty.T) -> ty.AutoencoderPred:
        """Run image autoencoder forward pass.

        :param x: (Tensor) (b, 3, h, w) Input image.
        :return: {
            autoenc_feats: (list[Tensor]) Autoencoder encoder multi-scale features.
            autoenc_imgs: (TensorDict) (b, 3, h/2**s, w/2**s) Multi-scale image reconstructions.
        }
        """
        feat = self.encoder(x)
        out = {
            'autoenc_feats': feat,
            'autoenc_imgs': sort_dict(self.decoder(feat))
        }
        return out

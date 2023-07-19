import torch
import torch.nn as nn
from torchvision.transforms import Compose

__all__ = ['load_midas_net']


def load_midas_net(name: str) -> tuple[nn.Module, Compose]:
    """Load a frozen pre-trained Midas network from the PyTorch Hub.
    From: https://arxiv.org/abs/1907.01341 & https://arxiv.org/abs/2103.13413

    Pre-processing: ImageNet standarization & resizing (provided by `tfm`).
    Pre-processing: Bicubic resizing.

    Training shapes: Images are resized such that the height/width are 384 or 512 and aspect ratio is retained.
    See https://github.com/isl-org/MiDaS/blob/1645b7e1675301fdfac03640738fe5a6531e17d6/midas/transforms.py#L48 for
    additional details

    Prediction:
    ```
    # Prediction is made in scaleless disparity.
    img: np.array (unit8)
    pred = net(tfm(img))
    ```

    :param name: (str) Model configuration to load.
    :return: (nn.Module, Any) Loaded model & pre-processing transforms.
    """
    net = torch.hub.load('intel-isl/MiDaS', name).eval()
    for p in net.parameters(): p.requires_grad = False

    tfms = torch.hub.load('intel-isl/MiDaS', "transforms")

    if 'DPT' in name: tfm = tfms.dpt_transform
    elif 'BEiT' in name: tfm = tfms.beit512_transform
    elif name == 'MiDaS_small': tfm = tfms.small_transform
    else: tfm = tfms.default_transform

    return net, tfm


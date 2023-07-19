from typing import Optional

import torch
import torch.nn as nn

from .newcrfs.networks.NewCRFDepth import NewCRFDepth
from .. import PATHS

__all__ = ['load_newcrfs_net']

SCENES = {'indoor', 'outdoor'}


def load_newcrfs_net(scene: str, max_depth: Optional[float] = None) -> nn.Module:
    """Load a frozen pre-trained NeWCRFs network.
    From https://arxiv.org/abs/2203.01502.

    Pre-processing: ImageNet standarization & resizing.

    Training shapes:
        - Outdoor: (Kitti) (352, 1216)
        - Indoor: (NYUD) (480, 640)

    Prediction:
    ```
    # Prediction is made in METRIC DEPTH, not disparity.
    img: torch.Tensor
    pred = net(ops.standardize(img))
    ```

    :param scene: (str) Model type to load. {`indoor`, `outdoor`}
    :param max_depth: (None|float) Maximum expected metric depth.
    :return: (nn.Module) Loaded network.
    """
    if scene not in SCENES: raise ValueError(f'Invalid NeWCRFs model. ({scene} vs. {SCENES})')

    max_depth = max_depth or (10 if scene == 'indoor' else 80)
    if max_depth <= 0: raise ValueError(f'Max depth must be a positive number. Got {max_depth}.')

    net = nn.DataParallel(NewCRFDepth(version='large07', inv_depth=False, max_depth=max_depth, pretrained=None))

    ckpt_file = PATHS['newcrfs_indoor'] if scene == 'indoor' else PATHS['newcrfs_outdoor']
    ckpt = torch.load(ckpt_file, map_location='cpu')
    net.load_state_dict(ckpt['model'])

    net = net.eval()
    for p in net.parameters(): p.requires_grad = False
    return net

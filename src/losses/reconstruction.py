import torch
import torch.nn as nn

import src.typing as ty
from src import register
from src.tools import ops
from . import DenseL1Error, DenseL2Error, PhotoError

__all__ = ['ReconstructionLoss']


@register(('img_recon', 'feat_recon', 'autoenc_recon'))
class ReconstructionLoss(nn.Module):
    """Class to compute the reconstruction loss when synthesising new views.

    Contributions:
        - Min reconstruction error: From Monodepth2 (https://arxiv.org/abs/1806.01260)
        - Static pixel automasking: From Monodepth2 (https://arxiv.org/abs/1806.01260)
        - Explainability mask: From SfM-Learner (https://arxiv.org/abs/1704.07813)
        - Uncertainty mask: From Klodt (https://openaccess.thecvf.com/content_ECCV_2018/papers/Maria_Klodt_Supervising_the_new_ECCV_2018_paper.pdf)

    :param loss_name: (str) Loss type to use.
    :param use_min: (bool) If `True`, take the final loss as the minimum across all available views.
    :param use_automask: (bool) If `True`, mask pixels where the original support image has a lower loss than the warped counterpart.
    :param mask_name: (None|str) Weighting mask used. {'explainability', 'uncertainty', None}
    """
    def __init__(self, loss_name: str = 'ssim', use_min: bool = False, use_automask: bool = False, mask_name: ty.U[str] = None):
        super().__init__()
        self.loss_name = loss_name
        self.use_min = use_min
        self.use_automask = use_automask
        self.mask_name = mask_name

        if self.mask_name not in {'explainability', 'uncertainty', None}:
            raise ValueError(f"Invalid mask type: {self.mask_name}")

        self._photo = {
            'ssim':  PhotoError(weight_ssim=0.85),
            'l1': DenseL1Error(),
            'l2': DenseL2Error(),
        }[self.loss_name]

        self._reduce = (lambda x: x.min(dim=1, keepdim=True)[0]) if self.use_min else \
            (lambda x: x.mean(dim=1, keepdim=True))

    def apply_mask(self, err: ty.T, mask: ty.N[ty.T] = None) -> ty.T:
        """Apply a weighting mask to a photometric loss error.

        :param err: (Tensor) (b, n, h, w) Photometric error to mask.
        :param mask: (None|ty.T) (b, n, h, w) Optional weighting mask to apply.
        :return: (Tensor) (b, n, h, w) The weighted photometric error.
        """
        if self.mask_name and mask is None: raise ValueError("Must provide a 'mask' when masking...")

        if self.mask_name == 'explainability': err *= mask
        elif self.mask_name == 'uncertainty':  err = err*(-mask).exp() + mask  # Reg to avoid max uncertainty.
        return err

    def apply_automask(self, err: ty.T, source: ty.T, target: ty.T, mask: ty.N[ty.T] = None) -> tuple[ty.T, ty.T]:
        """Compute and apply an automask based on the identity reconstruction error.

        :param err: (Tensor) (b, 1, h, w) The photometric error between target and warped support frames.
        :param target: (Tensor) (b, 3, h, w) Target image to reconstruct.
        :param source: (None|Tensor) (*n, b, 3, h, w) Original support images.
        :param mask: (None|Tensor) (b, n, h, w) Optional weighting mask for the photometric error.
        :return: (
            err: (Tensor) (b, 1, h, w) The automasked photometric error.
            automask: (Tensor) (b, 1, h, w) Boolean mask indicating valid pixels after automasking.
        )
        """
        err_static = self.compute_photo(source, target, mask=mask)  # (b, 1, h, w)
        err_static += ops.eps(err_static)*torch.randn_like(err_static)  # Break ties.

        err = torch.cat((err, err_static), dim=1)  # (b, 2, h, w)
        err, idxs = torch.min(err, dim=1, keepdim=True)
        automask = idxs == 0
        return err, automask

    def compute_photo(self, pred: ty.T, target: ty.T, mask: ty.N[ty.T] = None) -> ty.T:
        """Compute the dense photometric between multiple predictions and a single target.

        :param pred: (Tensor) (*n, b, 3, h, w) Synthesized warped support images.
        :param target: (Tensor) (b, 3, h, w) Target image to reconstruct.
        :param mask: (None|Tensor) (b, n, h, w) Optional weighting mask for the photometric error.
        :return: (Tensor) (b, 1, h, w) The reduced photometric error.
        """
        if pred.ndim == 4:  # Single support frame.
            err = self._photo(pred, target)
        else:  # Multiple support frames.
            target = target[None].expand_as(pred)  # (n, b, 3, h, w)
            err = self._photo(pred.flatten(0, 1), target.flatten(0, 1))  # (n*b, 1, h, w)
            err = err.squeeze(1).unflatten(0, pred.shape[:2]).permute(1, 0, 2, 3)  # (b, n, h, w)

        err = self.apply_mask(err, mask)
        err = self._reduce(err)  # (b, 1, h, w)
        return err

    def forward(self,
                pred: ty.T,
                target: ty.T,
                source: ty.N[ty.T] = None,
                mask: ty.N[ty.T] = None) -> ty.LossData:
        """Compute the reconstruction loss between two images.

        :param pred: (Tensor) (*n, b, 3, h, w) Synthesized warped support images.
        :param target: (Tensor) (b, 3, h, w) Target image to reconstruct.
        :param source: (None|Tensor) (*n, b, 3, h, w) Original support images.
        :param mask: (None|Tensor) (b, n, h, w) Optional weighting mask for the photometric error.
        :return: (
            loss: (Tensor) (,) Scalar loss.
            loss_dict: {
                (Optional) (If using automasking)
                automask: (Tensor) (b, 1, h, w) Boolean mask indicating valid pixels after automasking.
            }
        )
        """
        ld = {}
        err = self.compute_photo(pred, target, mask)  # (b, 1, h, w)

        if self.use_automask:
            if source is None: raise ValueError("Must provide the original 'source' images when automasking...")
            err, automask = self.apply_automask(err, source, target, mask)
            ld['automask'] = automask

        loss = err.mean()
        return loss, ld

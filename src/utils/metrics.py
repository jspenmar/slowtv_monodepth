"""Metrics used to track progress while training."""
import torch
from torchmetrics import Metric

import src.typing as ty

__all__ = ['MAE', 'RMSE', 'ScaleInvariant', 'AbsRel', 'SqRel', 'DeltaAcc']


_MODES = {'raw', 'log', 'inv'}


class BaseMetric(Metric):
    """Base class for depth estimation metrics."""
    higher_is_better = False
    full_state_update = False

    def __init__(self, mode: str = 'raw', **kwargs):
        super().__init__(**kwargs)
        if mode not in _MODES: raise ValueError(f"Invalid mode! ({mode} vs. {_MODES})")

        self.mode: str = mode
        self.sf: int = {'raw': 1, 'log': 100, 'inv': 1000}[self.mode]  # Scaling factor to align significant figures.

        self.add_state('metric', default=torch.tensor(0.), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')

    def _preprocess(self, input: ty.T, /):
        """Convert input into log-depth or disparity."""
        if self.mode == 'raw': pass
        elif self.mode == 'log': input = input.log()
        elif self.mode == 'inv': input = 1/input.clip(min=1e-3)
        return input

    def _compute(self, pred: ty.T, target: ty.T) -> ty.T:
        """Compute an error metric for a single pair.

        :param pred: (Tensor) (b, n) Predicted depth.
        :param target: (Tensor) (b, n) Target depth.
        :return: (Tensor) (b,) Computed metric.
        """
        raise NotImplementedError

    def update(self, pred: ty.T, target: ty.T) -> None:
        """Compute an error metric for a whole batch of predictions and update the state.

        :param pred: (Tensor) (b, n) Predicted depths masked with NaNs.
        :param target: (Tensor) (b, n) Target depths masked with NaNs.
        :return:
        """
        self.metric += self.sf * self._compute(self._preprocess(pred), self._preprocess(target)).sum()
        self.total += pred.shape[0]

    def compute(self) -> ty.T:
        """Compute the average metric given the current state."""
        return self.metric / self.total


class MAE(BaseMetric):
    """Compute the mean absolute error."""
    def _compute(self, pred: ty.T, target: ty.T) -> ty.T:
        return (pred - target).abs().nanmean(dim=1)


class RMSE(BaseMetric):
    """Compute the root mean squared error."""
    def _compute(self, pred: ty.T, target: ty.T) -> ty.T:
        return (pred - target).pow(2).nanmean(dim=1).sqrt()


class ScaleInvariant(BaseMetric):
    """Compute the scale invariant error."""
    def _compute(self, pred: ty.T, target: ty.T) -> ty.T:
        err = pred - target
        return (err.pow(2).nanmean(dim=1) - err.nanmean(dim=1).pow(2)).sqrt()


class AbsRel(BaseMetric):
    """Compute the absolute relative error."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sf = 100  # As %

    def _compute(self, pred: ty.T, target: ty.T) -> ty.T:
        return ((pred - target).abs() / target).nanmean(dim=1)


class SqRel(BaseMetric):
    """Compute the absolute relative squared error."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sf = 100  # As %

    def _compute(self, pred: ty.T, target: ty.T) -> ty.T:
        return ((pred - target).pow(2) / target.pow(2)).nanmean(dim=1)


class DeltaAcc(BaseMetric):
    """Compute the accuracy for a given error threshold."""
    higher_is_better = True

    def __init__(self, delta: float, **kwargs):
        super().__init__(**kwargs)
        if self.mode != 'raw': raise ValueError('DeltaAcc should only be computed using raw depths.')
        self.delta: float = delta
        self.sf = 100  # As %

    def _compute(self, pred: ty.T, target: ty.T) -> ty.T:
        thresh = torch.max(target/pred, pred/target)
        return (thresh < self.delta).nansum(dim=1) / thresh.nansum(dim=1)

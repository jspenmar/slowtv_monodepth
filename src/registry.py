"""Registry manager for networks, losses, data & predictors."""
import torch.optim.lr_scheduler as sched

import src.typing as ty
from src.utils import Timer, get_logger

__all__ = [
    'register',
    'NET_REG', 'LOSS_REG', 'DATA_REG', 'SCHED_REG', 'PRED_REG', 'DEC_REG',
    'trigger_nets', 'trigger_datas', 'trigger_losses', 'trigger_preds', 'trigger_decoders',
]

CLS = ty.TypeVar('T')
logger = get_logger('Registry')

NET_REG: ty.ModDict = {}
LOSS_REG: ty.ModDict = {}
DATA_REG: ty.DataDict = {}
PRED_REG: ty.PredDict = {}
DEC_REG: ty.ModDict = {}
SCHED_REG: ty.SchedDict = {
    'steplr': sched.StepLR,
    'exp': sched.ExponentialLR,
    'cos': sched.CosineAnnealingLR,
    'cos_warm': sched.CosineAnnealingWarmRestarts,
    'plateau': sched.ReduceLROnPlateau,
    'linear': sched.LinearLR,
}

# Collection of registries.
_REG: ty.DictDict = {
    'net': NET_REG,
    'loss': LOSS_REG,
    'data': DATA_REG,
    'pred': PRED_REG,
    'dec': DEC_REG,
}

# Patterns matching class name endings to registry types.
_NAME2TYPE: ty.StrDict = {
    'Net': 'net',
    'Loss': 'loss',
    'Reg': 'loss',
    'Dataset': 'data',
    'Pred': 'pred',
    'Predictor': 'pred',
    'Dec': 'dec',
    'Decoder': 'dec',
}


# noinspection PyUnresolvedReferences
def trigger_nets() -> None:
    """Trigger adding all networks to the registry."""
    with Timer(as_ms=True) as t: from src import networks
    logger.debug(f"Triggered registry networks in {t.elapsed}ms...")


# noinspection PyUnresolvedReferences
def trigger_datas() -> None:
    """Trigger adding all datasets to the registry."""
    with Timer(as_ms=True) as t: from src import datasets
    logger.debug(f"Triggered registry datasets in {t.elapsed}ms...")


# noinspection PyUnresolvedReferences
def trigger_losses() -> None:
    """Trigger adding all losses to the registry."""
    with Timer(as_ms=True) as t: from src import losses, regularizers
    logger.debug(f"Triggered registry losses in {t.elapsed}ms...")


# noinspection PyUnresolvedReferences
def trigger_preds() -> None:
    """Trigger adding all predictors to the registry."""
    with Timer(as_ms=True) as t: from src.core import predictors
    logger.debug(f"Triggered registry predictors in {t.elapsed}ms...")


# noinspection PyUnresolvedReferences
def trigger_decoders() -> None:
    """Trigger adding all predictors to the registry."""
    with Timer(as_ms=True) as t: from src.networks import decoders
    logger.debug(f"Triggered registry decoders in {t.elapsed}ms...")


def register(name: ty.U[str, tuple[str]], type: ty.N[str] = None, overwrite: bool = False) -> CLS:
    """Class decorator to build a registry of networks, losses & data available during training.

    Example:
    ```
    # Register using default naming conventions. See `_NAME2TYPE`.
    @register('my_net')
    class MyNet(nn.Module): ...

    # Register to specific type.
    @register('my_loss', type='loss')
    class MyClass(nn.Module): ...

    # Register multiple names for the same class.
    @register(('my_dataset1', 'my_dataset2'))
    class MyDataset(Dataset): ...
    ```

    :param name: (str|Sequence[str]) Key(s) used to access class in the registry.
    :param type: (None|str) Registry to use. If `None`, guess from class name. {None, net, loss, data, pred}
    :param overwrite: (bool) If `True`, overwrite class `name` in registry `type`.
    :return:
    """
    def _guess_type(cls: CLS) -> str:
        """Helper to identify registry `type` from class name."""
        try:
            return next(v for k, v in _NAME2TYPE.items() if cls.__name__.endswith(k))
        except StopIteration:
            raise ValueError(f"Class matched no known patterns. ({cls.__name__} vs. {set(_NAME2TYPE)})")

    def wrapper(cls: CLS) -> CLS:
        """Decorator adding `cls` to the specified registry."""

        # Ignore classes created in __main__ entrypoint to avoid duplication.
        if cls.__module__ == '__main__':
            logger.warning(f"Ignoring class '{cls.__name__}' created in the '__main__' module.")
            return cls

        ns = (name,) if isinstance(name, str) else name
        t = type or _guess_type(cls)
        if t not in _REG: raise TypeError(f"Invalid `type`. ({t} vs. {set(_REG)})")

        reg = _REG[t]
        for n in ns:
            if not overwrite and (tgt := reg.get(n)):
                raise ValueError(f"'{n}' already in '{t}' registry ({tgt} vs. {cls}). Set `overwrite=True` to overwrite.")

            logger.debug(f"Added '{n}' to the '{t}' registry...")
            reg[n] = cls
        return cls

    return wrapper

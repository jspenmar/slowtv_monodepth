"""Collection of miscellaneous logging, dict and plotting utilities."""
import logging

from matplotlib import pyplot as plt
from rich.logging import RichHandler

import src.typing as ty

__all__ = ['get_logger', 'flatten_dict', 'sort_dict', 'apply_cmap', 'set_logging_level']


def set_logging_level(level: str = 'warning') -> None:
    """Set logging level for all loggers."""
    level = getattr(logging, level.upper())
    logging.basicConfig(level=level, format='%(message)s', handlers=[RichHandler()])
    [logging.getLogger(name).setLevel(level) for name in logging.root.manager.loggerDict]
    logging.getLogger('torch').setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get `Logger` with specified `name`, ensuring it has only one handler (including parents)."""
    logger = logging.getLogger(name)
    logger.propagate = False  # Don't propagate to parents (avoid duplication)
    if not logger.handlers: logger.addHandler(RichHandler())  # Only add handlers once (avoid duplication)
    return logger


def flatten_dict(d: dict, /, parent: str = '', sep: str = '/') -> dict:
    """Flatten an arbitrary dict of dicts into a single dict. Keys are merged using `sep`."""
    return dict(_flatten_dict_gen(d, parent, sep))


def _flatten_dict_gen(d: ty.Any, /, parent: str, sep: str) -> tuple[str, ty.Any]:
    for k, v in d.items():
        k_new = parent+sep+k if parent else k
        if isinstance(v, dict): yield from flatten_dict(v, k_new, sep=sep).items()
        else: yield k_new, v


def sort_dict(d: dict, reverse: bool = False) -> dict:
    """Return a dict with sorted keys."""
    return {k: d[k] for k in sorted(d, reverse=reverse)}


def apply_cmap(arr: ty.A, /, cmap: str = 'turbo', vmin: ty.N[float] = None, vmax: ty.N[float] = None) -> ty.A:
    """Apply a matplotlib colormap to an image.

    :param arr: (NDArray) (*) Array of any shape to map.
    :param cmap: (str) Matplotlib colormap name.
    :param vmin: (None|float) Minimum value to use when normalizing. If `None` use `input.min()`.
    :param vmax: (None|float) Maximum value to use when normalizing. If `None` use `input.max()`.
    :return (NDArray) (*, 3) The colormapped array, where each original value has an assigned RGB value.
    """
    vmin = arr.min() if vmin is None else vmin  # Explicit `None` check to avoid issues with 0
    vmax = arr.max() if vmax is None else vmax

    arr = arr.clip(vmin, vmax)
    arr = (arr - vmin) / (vmax - vmin + 1e-5)  # Normalize [0, 1]

    arr = plt.get_cmap(cmap)(arr)[..., :-1]  # Remove alpha
    return arr

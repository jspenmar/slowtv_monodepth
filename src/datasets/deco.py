import random
from functools import wraps

import src.typing as ty
from src.utils import opt_args_deco

__all__ = ['validated_init', 'retry_new_on_error']


def validated_init(__init__: ty.Callable):
    """Decorator to ensure a BaseDataset child always calls argument validation after init."""
    @wraps(__init__)
    def wrapper(self, *args, **kwargs) -> None:
        self.logger.info(f"Creating '{self.__class__.__qualname__}'...")
        __init__(self, *args, **kwargs)
        self.log_args()
        self.validate_args()
    return wrapper


@opt_args_deco
def retry_new_on_error(__getitem__: ty.Callable,
                       exc: ty.U[BaseException, ty.S[BaseException]] = Exception,
                       silent: bool = False,
                       max: ty.N[int] = None,
                       use_blacklist: bool = False) -> ty.Callable:
    """Decorator to wrap a BaseDataset __getitem__ function and retry a different item if there is an error.

    The idea is to provide a way of ignoring missing/corrupt data without having to blacklist files,
    change number of items and do "hacky" workarounds.
    Obviously, the less data we have, the less sense this decorator makes, since we'll start duplicating more
    and more items (although if we're augmenting our data, it shouldn't be too tragic).
    Obviously as well, for debugging/evaluation it probably makes more sense to disable this decorator.

    NOTE: This decorator assumes we follow the BaseDataset format
        - We return three dicts (x, y, m)
        - Errors are logged in meta['errors']
        - A 'log_timings' flag indicates the presence of a 'MultiLevelTimer' in self.timer

    :param __getitem__: (ty.Callable) Dataset `__getitem__` method to decorate.
    :param exc: (Exception|tuple[Exception]) Expected exceptions to catch and retry on.
    :param silent: (bool) If `False`, log error info to `meta`.
    :param max: (None|int) Maximum number of retries for a single item.
    :param use_blacklist: (bool) If `True`, keep a list of items to avoid.
    :return: (BatchData) Batch returned by `__getitem__`.
    """
    n = 0
    blacklist = set()

    # Multiple exceptions must be provided as tuple
    exc = exc or tuple()
    if isinstance(exc, list): exc = tuple(exc)

    @wraps(__getitem__)
    def wrapper(self, item: int) -> ty.BatchData:
        nonlocal n

        try:
            x, y, m = __getitem__(self, item)
            if not silent and 'errors' not in m: m['errors'] = ''
        except exc as e:
            n += 1
            if max and n >= max: raise RuntimeError('Exceeded max retries when loading dataset item...')

            if use_blacklist: blacklist.add(item)
            if self.log_time: self.timer.reset()

            new = item
            while new == item or new in blacklist:  # Force new item
                new = random.randrange(len(self))

            x, y, m = wrapper(self, new)
            if not silent: m['errors'] += f'{" - " if m["errors"] else ""}{(item, e)}'

        n = 0  # Reset!
        return x, y, m
    return wrapper

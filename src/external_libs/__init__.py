# noinspection PyUnresolvedReferences
from src.paths import MODEL_PATHS as PATHS  # Used in __all__ and submodules.
from .databases import *
from .midas import *
from .newcrfs import *

__all__ = (
    databases.__all__ +
    midas.__all__ +
    newcrfs.__all__ +
    ['PATHS']
)

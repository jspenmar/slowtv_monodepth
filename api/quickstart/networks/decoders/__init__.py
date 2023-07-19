from .dpt import *
from .monodepth import *

DECODERS = {
    'dpt': DptDecoder,
    'monodepth': MonodepthDecoder,
}

__all__ = (
        dpt.__all__ +
        monodepth.__all__
)

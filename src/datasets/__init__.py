from .base import *
from .base_mde import *

from .ddad import *
from .diode import *
from .kitti_raw import *
from .kitti_raw_lmdb import *
from .mannequin import *
from .mannequin_lmdb import *
from .mapfreereloc import *
from .nyud import *
from .sintel import *
from .slow_tv import *
from .slow_tv_lmdb import *
from .syns_patches import *
from .tum import *

__all__ = (
    base.__all__ +
    base_mde.__all__ +
    ddad.__all__ +
    diode.__all__ +
    kitti_raw.__all__ +
    kitti_raw_lmdb.__all__ +
    mannequin.__all__ +
    mannequin_lmdb.__all__ +
    mapfreereloc.__all__ +
    nyud.__all__ +
    sintel.__all__ +
    slow_tv.__all__ +
    slow_tv_lmdb.__all__ +
    syns_patches.__all__ +
    tum.__all__
)

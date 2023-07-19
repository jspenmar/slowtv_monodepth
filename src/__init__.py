import logging
import warnings
from pathlib import Path

# Remove warnings.
from tqdm import TqdmExperimentalWarning
warnings.filterwarnings(action='ignore', category=TqdmExperimentalWarning)
warnings.filterwarnings(action='ignore', category=UserWarning, module='mmcv')

# Downloading pretrained models on cluster can fail otherwise.
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

# Change torch hub default dir.
import torch.hub
torch.hub.set_dir(Path(__file__).parent.parent/'models'/'torch')

from .utils import set_logging_level, get_logger, Timer
set_logging_level()
LOGGER = get_logger('MDB')

with Timer(as_ms=True) as t: from .registry import *
logging.debug(f"Loaded 'registry' in {t.elapsed}ms")

with Timer(as_ms=True) as t: from .paths import *
logging.debug(f"Loaded 'paths' in {t.elapsed}ms")

__all__ = [
    registry.__all__ +
    paths.__all__ +
    ['LOGGER']
]

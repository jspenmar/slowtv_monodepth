import inspect
from abc import ABC, abstractmethod
from contextlib import nullcontext

import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

import src.typing as ty
from src.tools import ops
from src.utils import MultiLevelTimer, default_collate, delegates, get_logger, io
from .deco import retry_new_on_error, validated_init

__all__ = ['BaseDataset']


class BaseDataset(ABC, Dataset):
    """Base dataset class that all others should inherit from.

    The idea is to provide a common structure and data format. Additionally, provide some nice functionality and
    automation for the more boring stuff. Datasets are defined as providing the following dicts for each item:
        - x: Inputs to the network (typically `imgs`).
        - y: Additional data required for loss computation (e.g. `labels`) or logging (e.g. non-augmented images).
        - m: Metadata for the given item, typically for logging.

    BaseDataset will automatically add the following fields to `m`:
        - items: Item number (i.e. argument to  `__getitem__`).
        - items_original: If `randomize`, original item number.
        - errors: If `retry_exc` and NOT `silent`, log caught exception messages.
        - aug: If `use_aug`, child class should add a list of the aug performed.
        - data_timer: Timing information for current item.

    Additional features/utilities provided include:
        - A logger to be used for logging.
        - A timer which, if enabled, times load/augment for an item. Can also be used in the child class.
        - Automatic retrying if the current item fails to load. This aims to replace "hacky"
          methods for manually filtering/blacklisting items, whilst being easy to enable & customize.
        - Functionality to limit the dataset length via the `max_len` argument.
        - Ability to randomize dataset loading order. (Should probably only be used when combined with `max_len`).
        - Argument validation via `validate_args`.
        - Tools for visualizing/playing the dataset to inspect and sanity check it.

    Loading pipeline:
        The process of loading items from the dataset is separated into four components.
            - Loading: Performs the actual loading from the dataset & arranges into (x, y, m).
            - Augmentation: (Optional) Randomly augments the loaded data.
            - Transform: Fixed pre-processing required for training (typically, image normalization).
            - ToTorch: (Optional) Convert all loaded items into torch tensors and permute into channel-first format.

        Loading works based on a `datum` system, where the user provides a list of data types to load. Child classes are
        required to implement functions to load each of these data types (as `load_<data_type>) and add any required
        metadata (`add_metadata`).

        Classes are also required to provide a class attribute `VALID_DATUM` that lists the valid datatypes that can
        be loaded. Additionally, `items_data` should be a list containing the data required to load each item. This
        provides the default implementation of `len` and `num_items`.

    Batch collating:
        The dataset also provides a `collate_fn` that should be forwarded when creating a DataLoader from a dataset.
        The default implementation falls back on the PyTorch default, but can be overriden if necessary.
        >>> ds = MyCustomDataset(mode='train', datum=['image', 'label'])
        >>> dl = DataLoader(ds, batch_size=4, num_workers=4, collate_fn=ds.collate_fn)

    Automatic retrying:
        Functionality to ignore loading exceptions and load a random item is provided on a per class (rather than per
        instance) basis. This is specified when declaring the class:
        >>> class MyCustomDataset(BaseDataset, retry_exc=(FileNotFoundError, KeyError), max_retries=10): pass

    Paramers:
    :param datum: (list[str]) Datatypes to load.
    :param use_aug: (bool) If `True`, call 'self.augment' during __getitem__.
    :param as_torch: (bool) If `True`, convert (x, y, meta) to torch.
    :param max_len: (None|int) Max number of items to load. Combine with `randomize` to limit epoch duration.
    :param randomize: (bool) If `True`, randomize the item number when loading.
    :param log_time: (bool) If `True`, log time taken to load/augment each item.

    Attributes:
    :attr VALID_DATUM: (REQUIRED) (set[str]) Class attribute representing valid datatypes that can be loaded.
    :attr items_data: (REQUIRED) (list[Any]) List of data required to load each dataset item (e.g. image filenames).
    :attr logger: (Logger) Logger with parent CogvisDataset to use for logging.
    :attr timer: (MultiLevelTimer) If 'log_timings', timer to use for timing blocks.

    Methods:
    :method __init_subclass__: Subclass initializer to create logger and wrap __getitem__ and __init__.
    :method __repr__: String representation containing parameters required to initialize dataset.
    :method validate_args: (OVERRIDE) Error checking for provided dataset configuration.
    :method collate_fn: Classmethod to collate multiple dataset items into a batch.
    :method __len__: Number of dataset items, adjusted by `max_len`. Do not modify!
    :method num_items: (OVERRIDE) Raw number of dataset items, based on `items_data`.
    :method is_valid: Helper to determine if the provided datatype is legal for this dataset.
    :method has: Helper to check if a datatype should be loaded by this dataset.
    :method get_load_fn: Retrieve the corresponding loading function for the provided datatype.
    :method __getitem__: Dataset item loading pipeline as (Load -> Augment -> Transform -> ToTorch). Do not modify!
    :method load: Load a single dataset item and arrange into (x, y, m) dicts.
    :method load_<datatype>: (REQUIRED) Loading function for each datatype provided in `datum`.
    :method add_metadata: (OVERRIDE) Add required item metadata based on the item data.
    :method augment: (OVERRIDE) Augment a dataset item.
    :method transform: (OVERRIDE) Fixed dataset item transforms to apply.
    :method to_torch: (OVERRIDE) Convert dataset to torch Tensors.
    :method create_axs: (OVERRIDE) Create axes required for plotting.
    :method show: (REQUIRED) Show a single dataset item.
    :method play: Iterate through dataset and display.
    """
    _tagged = False  # Ensure argument checking is only applied once.

    # INIT
    # -----------------------------------------------------------------------------
    def __init__(self,
                 datum: ty.S[str] = None,
                 use_aug: bool = False,
                 as_torch: bool = True,
                 max_len: ty.N[int] = None,
                 randomize: bool = False,
                 log_time: bool = True):
        self.datum = datum or []
        self.as_torch = as_torch
        self.use_aug = use_aug
        self.log_time = log_time
        self.max_len = max_len
        self.randomize = randomize

        self.items_data = None  # Must be overridden.

        if isinstance(self.VALID_DATUM, str): self.VALID_DATUM = set(self.VALID_DATUM.split())
        if isinstance(self.datum, str): self.datum = self.datum.split()

        # Timer setup - 'nullcontext' allows for a cleaner 'getitem' without too many conditionals
        self.timer = MultiLevelTimer(name=self.__class__.__qualname__, as_ms=True, precision=4) if self.log_time \
            else nullcontext

    def __init_subclass__(cls,
                          retry_exc: ty.N[ty.U[Exception, tuple[Exception]]] = None,
                          silent: bool = False,
                          max_retries: int = 10,
                          use_blacklist: bool = False,
                          **kwargs):
        """Subclass initializer to create logger and wrap __getitem__ and __init__.

        :param retry_exc: (None|Exception|tuple[Exception]) Exceptions to ignore and retry a different item.
        :param silent: (bool) If `False`, log error info to `meta`.
        :param max_retries: (None|int) Maximum number of retries for a single item.
        :param use_blacklist: (bool) If `True`, keep a list of items to avoid.
        :param kwargs: (dict) Kwargs required by parent classes (typically none required).
        """
        super().__init_subclass__(**kwargs)
        cls.logger = get_logger(f'BaseDataset.{cls.__qualname__}')
        cls.__init__ = delegates(cls.__base__.__init__)(cls.__init__)  # Replace kwargs in child signature.

        # Argument checking should not be applied to abstract datasets, as they likely won't be fully initialized.
        if not inspect.isabstract(cls) and not cls._tagged:
            cls._tagged = True
            cls.__init__ = validated_init(cls.__init__)

        cls.__getitem__ = retry_new_on_error(
            cls.__getitem__,
            exc=retry_exc,
            silent=silent,
            max=max_retries,
            use_blacklist=use_blacklist,
        )

    def __repr__(self) -> str:
        """String representation containing parameters required to initialize dataset."""
        sig = inspect.signature(self.__init__)
        kw = {k: getattr(self, k) for k in sig.parameters if hasattr(self, k)}
        kw = ', '.join(f'{k}={v}' for k, v in kw.items())
        return f'{self.__class__.__qualname__}({kw})'

    def validate_args(self) -> None:
        """Error checking for provided dataset configuration. Should always call parent `validate_args`."""
        if self.__class__.__len__ is not BaseDataset.__len__:
            raise TypeError('Child datasets of `BaseDataset` should not override the `__len__` method, '
                            'as this is used to implement the `max_len` feature. '
                            'Classes should instead override the default implementation of `num_items` if required!')

        if self.items_data is None: raise NotImplementedError('Dataset must provide `items_data` to use for loading...')
        if not self.datum: raise ValueError('Must provide `datum` to load!')

        ds = [d for d in self.datum if not self.is_valid(d)]
        if any(ds): raise ValueError(f'Invalid data types. ({ds} vs. {self.VALID_DATUM})')

        fns = [f for d in self.datum if not hasattr(self, f := f'load_{d}')]
        if any(fns): raise NotImplementedError(f'Missing data loading functions: {fns}')

    def log_args(self) -> None:
        """Log creation arguments. Extend in child classes if required."""
        self.logger.debug(f"Loading datum: {self.datum}...")

        if self.use_aug: self.logger.debug("Applying dataset augmentations...")
        if self.log_time: self.logger.debug("Logging dataset loading times...")
        if self.max_len: self.logger.debug(f"Restricting dataset to {self.max_len} items...")
        if self.randomize: self.logger.debug("Randomizing dataset item number loading...")
    # -----------------------------------------------------------------------------

    # COLLATING
    # -----------------------------------------------------------------------------
    @classmethod
    def collate_fn(cls, batch: ty.S[ty.BatchData]) -> ty.BatchData:
        """Classmethod to collate multiple dataset items into a batch. Default uses PyTorch default collating."""
        batch = io.tmap(default_collate, zip(*batch))
        return batch
    # -----------------------------------------------------------------------------

    # LEN
    # -----------------------------------------------------------------------------
    @ty.final
    def __len__(self) -> int:
        """Number of dataset items, adjusted by `max_len`. Do not modify!"""
        return min(self.num_items(), self.max_len) if self.max_len else self.num_items()

    def num_items(self) -> int:
        """Raw number of dataset items, based on `items_data`. Override if required."""
        return len(self.items_data)
    # -----------------------------------------------------------------------------

    # DATUM
    # -----------------------------------------------------------------------------
    @property
    @abstractmethod
    def VALID_DATUM(self) -> ty.U[str, set]:
        """Set of valid datatypes that can be loaded. Must be provided as a class attribute."""

    def is_valid(self, data_type: str) -> bool:
        """Helper to determine if the provided datatype is legal for this dataset."""
        return data_type in self.VALID_DATUM

    def has(self, data_type: str) -> bool:
        """Helper to check if a datatype should be loaded by this dataset."""
        if not self.is_valid(data_type): raise ValueError(f'Invalid data type. ({data_type} vs. {self.VALID_DATUM})')
        return data_type in self.datum

    def get_load_fn(self, data_type: str) -> ty.Callable:
        """Retrieve the corresponding loading function for the provided datatype."""
        if not self.is_valid(data_type): raise ValueError(f'Invalid data type. ({data_type} vs. {self.VALID_DATUM})')
        return getattr(self, f'load_{data_type}')
    # -----------------------------------------------------------------------------

    # LOADING
    # -----------------------------------------------------------------------------
    def __getitem__(self, item: int) -> ty.BatchData:
        """Dataset item loading pipeline as (Load -> Augment -> Transform -> ToTorch). Do not modify!"""
        if item >= len(self): raise IndexError
        if self.randomize:
            iitem, item = item, torch.randint(self.num_items(), ()).item()
            self.logger.debug(f"Randomized {iitem} into {item}...")

        self.logger.debug(f"Loading item {item}...")
        batch = x, y, m = {}, {}, {'items': str(item)}
        if self.randomize: m['items_original'] = str(iitem)
        if self.use_aug: m['augs'] = ''

        with self.timer('Total'):
            with self.timer('Load'): batch = self.load(item, batch)

            if self.use_aug:
                with self.timer('Augment'): batch = self.augment(batch)

            with self.timer('Transform'): batch = self.transform(batch)

            # NOTE: `as_torch` returns a copy of the batch. As such, we need to reassign `x, y, m`.
            if self.as_torch:
                with self.timer('ToTorch'): x, y, m = batch = self.to_torch(batch)

        if self.log_time:
            m['timer_data'] = self.timer.copy()
            self.logger.debug(str(self.timer))
            self.timer.reset()
        return batch

    def load(self, item: int, batch: ty.BatchData) -> ty.BatchData:
        """Load a single dataset item and arrange into (x, y, m) dicts. Should not require overriding."""
        data = self.items_data[item]
        batch = self.add_metadata(data, batch)
        for d in self.datum:
            self.logger.debug(f'Loading "{d}"...')
            with self.timer(d.capitalize()): batch = self.get_load_fn(d)(data, batch)
        return batch

    def add_metadata(self, data: ty.Any, batch: ty.BatchData) -> ty.BatchData:
        """Add required item metadata based on the item data. Override if required."""
        return batch

    def augment(self, batch: ty.BatchData) -> ty.BatchData:
        """Augment a dataset item. Override if required."""
        return batch

    def transform(self, batch: ty.BatchData) -> ty.BatchData:
        """Fixed dataset item transforms to apply. Override if required."""
        return batch

    def to_torch(self, batch: ty.BatchData) -> ty.BatchData:
        """Convert dataset to torch Tensors. Should not require overriding."""
        return ops.to_torch(batch)
    # -----------------------------------------------------------------------------

    # DISPLAYING
    # -----------------------------------------------------------------------------
    def create_axs(self) -> ty.Axes:
        """Create axes required for displaying."""
        _, ax = plt.subplots()
        return ax

    @abstractmethod
    def show(self, batch: ty.BatchData, axs: ty.Axes) -> None:
        """Show a single dataset item."""

    def play(self,
             fps: float = 30,
             skip: int = 1,
             reverse: bool = False,
             fullscreen: bool = False,
             axs: ty.N[ty.Axes] = None,
             title: ty.N[ty.Callable[[int, ty.BatchData], str]] = None) -> None:
        """Iterate through dataset and display.

        :param fps: (int) Frames per second. (Likely not required).
        :param skip: (int) Gap between items.
        :param reverse: (bool) If `True`, iterate though items in reverse order.
        :param fullscreen: (bool) If `True` make figure fullscreen.
        :param axs: (None|Axes) Axes to display items on.
        :param title: (None|Callable) Optional function that accepts the item number and batch to return a title.
        :return:
        """
        if self.as_torch: raise ValueError('Dataset must not be in torch format when playing.')

        axs = self.create_axs() if axs is None else axs
        fig = plt.gcf()
        if fullscreen: fig.canvas.manager.full_screen_toggle()

        if title is None: title = lambda i, b: str(i)

        items = range(len(self)-1, 0, -skip) if reverse else range(0, len(self), skip)
        for i in tqdm(items):
            axs.cla() if isinstance(axs, plt.Axes) else [ax.cla() for ax in axs.flatten()]
            batch = self[i]
            self.show(batch, axs)
            fig.suptitle(title(i, batch))
            plt.pause(1/fps)
        plt.show(block=False)
    # -----------------------------------------------------------------------------

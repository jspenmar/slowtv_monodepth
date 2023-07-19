from itertools import chain, zip_longest

from torch.utils.data import DataLoader, DistributedSampler

import src.typing as ty

__all__ = ['ConcatDataLoader']


class ConcatDataLoader:
    """Concatenate multiple DataLoaders in a round-robin manner.
    Example:
        dl1 = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        dl2 = ['a', 'b', 'c']
        dl3 = [0.1, 0.2, 0.3, 0.4. 0.5]

        [0, 'a', 0.1, 1, 'b', 0.2, 3, 'c', 0.3, 4, 0.4, 5, 0.5, 6, 7, 8]

    :param dls: (Sequence[DataLoader]) List of dataloaders to combine.
    """
    def __init__(self, dls: ty.S[DataLoader]):
        self.dls = dls
        print(f'-> Created Concat DataLoader with lengths: {[len(dl) for dl in self.dls]}')

    def __len__(self) -> int:
        """Number of items across all dataloaders."""
        return sum(map(len, self.dls))

    def __iter__(self) -> ty.BatchData:
        """Iterate over dataloaders in a round-robin manner."""
        yield from (i for i in chain.from_iterable(zip_longest(*self.dls)) if i is not None)

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch number. Required for DitributedSampler to randomize samples across multiple GPUs."""
        [dl.sampler.set_epoch(epoch) for dl in self.dls if isinstance(dl.sampler, DistributedSampler)]

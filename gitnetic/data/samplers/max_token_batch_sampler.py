import math
import random
from typing import Iterator, List

import numpy as np

from gitnetic.data import IndexedDataset
from gitnetic.utils import lazy_groups_of

from .batch_sampler import BatchSampler


class MaxTokensBatchSampler(BatchSampler):
    data_source: IndexedDataset

    def __init__(
        self,
        data_source: IndexedDataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
    ) -> None:
        super().__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[List[int]]:
        sorted_indices = np.argsort(self.data_source.sizes)
        batches = []
        for group in lazy_groups_of(sorted_indices, self.batch_size):
            batch_indices = list(group)
            if self.drop_last and len(batch_indices) < self.batch_size:
                continue
            batches.append(batch_indices)
        if self.shuffle:
            random.shuffle(batches)
        for batch in batches:
            yield batch

    def __len__(self) -> int:
        batch_count_float = len(self.data_source) / self.batch_size
        if self.drop_last:
            return math.floor(batch_count_float)
        return math.ceil(batch_count_float)
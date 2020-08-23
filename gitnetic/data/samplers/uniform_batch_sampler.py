import math
import random
from typing import Iterator, List, Optional

import numpy as np

from gitnetic.data.indexed_dataset import IndexedDatasetMixin
from gitnetic.data.samplers import BatchSampler
from gitnetic.utils import lazy_groups_of


class UniformBatchSampler(BatchSampler):
    data_source: IndexedDatasetMixin

    def __init__(
        self,
        data_source: IndexedDatasetMixin,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
    ) -> None:
        super().__init__(data_source, batch_size, shuffle, drop_last)

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


class UniformMaxTokensBatchSampler(BatchSampler):
    data_source: IndexedDatasetMixin

    def __init__(
        self,
        data_source: IndexedDatasetMixin,
        max_tokens: int,
        shuffle: bool = True,
        drop_last: bool = False,
    ) -> None:
        super().__init__(data_source, 0, shuffle, drop_last)
        self.max_tokens = max_tokens
        self._length: Optional[int] = None

    def __iter__(self) -> Iterator[List[int]]:
        sorted_indices = np.argsort(self.data_source.sizes)
        sorted_indices = sorted_indices.tolist()
        batches = []

        batch_indices = []
        batch_tokens = 0
        current_index = 0
        while current_index < len(sorted_indices):
            index = sorted_indices[current_index]
            batch_indices.append(index)
            batch_tokens += self.data_source.sizes[index]
            current_index += 1

            if batch_tokens >= self.max_tokens:
                batches.append(batch_indices)
                batch_indices = []
                batch_tokens = 0
            elif not sorted_indices and not self.drop_last:
                batches.append(batch_indices)

        if self.shuffle:
            random.shuffle(batches)

        for batch in batches:
            self.batch_size = len(batch)
            yield batch

    def __len__(self) -> int:
        if self._length is None:
            batches = list(iter(self))
            self._length = len(batches)
        return self._length
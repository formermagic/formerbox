from typing import Iterator, List

from torch.utils.data import Dataset, Sampler


class BatchSampler(Sampler):
    data_source: Dataset

    def __init__(self, data_source: Dataset) -> None:
        super().__init__(data_source)
        self.data_source = data_source

    def __iter__(self) -> Iterator[List[int]]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

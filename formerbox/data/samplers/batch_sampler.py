from typing import Iterator, List, Sized

from torch.utils.data import Dataset, Sampler


class BatchSampler(Sampler):
    def __init__(
        self,
        data_source: Dataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
    ) -> None:
        # pylint: disable=isinstance-second-argument-not-valid-type
        assert isinstance(data_source, Sized)

        super().__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    @classmethod
    def from_batch_sampler(cls, batch_sampler: "BatchSampler") -> "BatchSampler":
        return cls(
            batch_sampler.data_source,
            batch_sampler.batch_size,
            batch_sampler.shuffle,
            batch_sampler.drop_last,
        )

    def __iter__(self) -> Iterator[List[int]]:
        raise NotImplementedError()

    def __len__(self) -> int:
        raise NotImplementedError()

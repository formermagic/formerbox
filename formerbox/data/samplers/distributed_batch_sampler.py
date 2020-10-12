from typing import Iterator, List, Optional

from torch.utils.data import Dataset, DistributedSampler

from formerbox.data.samplers import BatchSampler


class IntDataset(Dataset):
    def __init__(self, data_source: List[int]) -> None:
        self.data_source = data_source

    def __getitem__(self, index: int) -> int:
        return self.data_source[index]

    def __len__(self) -> int:
        return len(self.data_source)


class DistributedBatchSampler(BatchSampler):
    """ `BatchSampler` wrapper that distributes across each batch multiple workers."""

    def __init__(
        self,
        batch_sampler: BatchSampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
    ) -> None:
        super().__init__(
            batch_sampler.data_source,
            batch_sampler.batch_size,
            batch_sampler.shuffle,
            batch_sampler.drop_last,
        )

        self.batch_sampler = batch_sampler
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[List[int]]:
        for batch in self.batch_sampler:
            distributed_sampler = DistributedSampler(
                dataset=IntDataset(batch),
                num_replicas=self.num_replicas,
                rank=self.rank,
                shuffle=self.shuffle,
            )

            yield list(distributed_sampler)

    def __len__(self) -> int:
        return len(self.batch_sampler)

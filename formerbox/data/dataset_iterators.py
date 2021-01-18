import logging
from typing import Any, Dict, List, Optional, Text

from formerbox.data.indexed_dataset import IndexedDatasetBase
from formerbox.data.samplers import (
    BatchSampler,
    UniformBatchSampler,
    UniformMaxTokensBatchSampler,
)
from torch import Tensor
from torch.utils.data import Dataset
from transformers import DataCollator

logger = logging.getLogger(__name__)


class DatasetIterator(Dataset):
    # pylint: disable=too-many-arguments
    def __init__(
        self,
        dataset: IndexedDatasetBase,
        collator: DataCollator,
        max_tokens: Optional[int] = None,
        batch_size: Optional[int] = None,
        shuffle: bool = True,
        drop_last: bool = False,
    ) -> None:
        self.dataset = dataset
        self.collator = collator
        self.batch_sampler = self.make_batch_sampler(
            dataset, max_tokens, batch_size, shuffle, drop_last
        )

        # read the sampled index batches
        self.index_batches = list(iter(self.batch_sampler))

    def __len__(self) -> int:
        return len(self.index_batches)

    def __getitem__(self, index: int) -> Dict[Text, Tensor]:
        index_batch = self.index_batches[index]
        input_ids = [self.dataset[idx] for idx in index_batch]
        return self.collator(input_ids)

    @staticmethod
    def make_batch_sampler(
        dataset: IndexedDatasetBase,
        max_tokens: Optional[int],
        batch_size: Optional[int],
        shuffle: bool,
        drop_last: bool,
    ) -> BatchSampler:
        if max_tokens is None and batch_size is None:
            raise ValueError(
                "Unable to prepare a batch sampler."
                " You must pass either a `batch_size`"
                " or a `max_tokens` argument."
            )

        batch_sampler: BatchSampler
        if max_tokens is not None:
            batch_sampler = UniformMaxTokensBatchSampler(
                data_source=dataset,
                max_tokens=max_tokens,
                shuffle=shuffle,
                drop_last=drop_last,
            )
        else:
            assert batch_size is not None
            batch_sampler = UniformBatchSampler(
                data_source=dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=drop_last,
            )

        return batch_sampler

    def collate_fn(self, samples: List[Any]) -> Any:
        assert len(samples) == 1
        return samples[0]

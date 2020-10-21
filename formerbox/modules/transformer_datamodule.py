import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Text, Type, Union

import torch
from formerbox.common.dataclass_argparse import DataclassBase
from formerbox.common.has_params import HasParsableParams
from formerbox.common.registrable import Registrable
from formerbox.data.dataset_iterators import DatasetIterator
from formerbox.data.indexed_dataset import IndexedDatasetBase
from formerbox.utils import path_to_posix
from pytorch_lightning import LightningDataModule
from pytorch_lightning.core.datamodule import _DataModuleWrapper
from torch.utils.data import DataLoader, Dataset
from transformers import (
    DataCollator,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from typing_extensions import _ProtocolMeta  # type: ignore

Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

logger = logging.getLogger(__name__)


class DataLoadingMixin:
    def __init__(
        self,
        max_tokens: Optional[int],
        batch_size: Optional[int],
        num_workers: int,
    ) -> None:
        super().__init__()
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.num_workers = num_workers

    def get_dataset_itr(
        self,
        dataset: IndexedDatasetBase,
        collator: DataCollator,
        shuffle: bool,
        drop_last: bool,
    ) -> DatasetIterator:
        dataset_itr = DatasetIterator(
            dataset,
            collator=collator,
            max_tokens=self.max_tokens,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
        )

        return dataset_itr


class _MetaDataModule(_ProtocolMeta, _DataModuleWrapper):
    """Implements both meta classes to avoid TypeError exceptions."""


class TransformerDataModule(
    DataLoadingMixin,
    LightningDataModule,
    Registrable,
    HasParsableParams,
    metaclass=_MetaDataModule,
):
    @dataclass
    class Params(DataclassBase):
        train_data_prefix: Text = field(
            metadata={"help": "A prefix path for the train dataset file."}
        )
        val_data_prefix: Text = field(
            metadata={"help": "A prefix path for the validation dataset file."}
        )
        batch_size: Optional[int] = field(
            default=None,
            metadata={"help": "A number of instances/sentences in a batch."},
        )
        max_tokens: Optional[int] = field(
            default=None,
            metadata={"help": "A number of tokens in a batch."},
        )
        num_workers: int = field(
            default=1,
            metadata={"help": "A number of workers for data loading."},
        )

    params: Params
    params_type: Type[Params] = Params

    def __init__(self, tokenizer: Tokenizer, params: Params) -> None:
        super().__init__(params.max_tokens, params.batch_size, params.num_workers)

        self.tokenizer = tokenizer
        self.params = params

        self.train_dataset: Optional[IndexedDatasetBase] = None
        self.train_iterator: Optional[Dataset] = None
        self.val_dataset: Optional[IndexedDatasetBase] = None
        self.val_iterator: Optional[Dataset] = None

        self.collator = DataCollatorForLanguageModeling(self.tokenizer)  # type: ignore

    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs  # no data to download

    def transfer_batch_to_device(self, batch: Any, device: torch.device) -> Any:
        del device  # lightning should have already moved a batch to the device
        return batch

    def setup(self, stage: Optional[Text] = None) -> None:
        del stage  # we don't use `stage` to build a dataloader

        # prepare a train dataset iterator
        train_path = path_to_posix(self.params.train_data_prefix)
        self.train_dataset = IndexedDatasetBase.from_file(train_path)
        self.train_iterator = self.get_dataset_itr(
            self.train_dataset, collator=self.collator, shuffle=True, drop_last=False
        )

        # prepare a validation dataset iterator
        val_path = path_to_posix(self.params.val_data_prefix)
        self.val_dataset = IndexedDatasetBase.from_file(val_path)
        self.val_iterator = self.get_dataset_itr(
            self.val_dataset, collator=self.collator, shuffle=False, drop_last=False
        )

    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        del args, kwargs  # use initialized properties to make a dataloader
        assert self.train_iterator is not None
        return DataLoader(self.train_iterator, num_workers=self.num_workers)

    def val_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        del args, kwargs  # use initialized properties to make a dataloader
        assert self.val_iterator is not None
        return DataLoader(self.val_iterator, num_workers=self.num_workers)

    def test_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        del args, kwargs  # use initialized properties to make a dataloader
        raise NotImplementedError()

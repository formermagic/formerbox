from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, List, Optional, Text, Tuple, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning import LightningDataModule, LightningModule
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AdamW,
    DataCollator,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from gitnetic.common.from_args import FromArgs
from gitnetic.data.dataset_iterators import DatasetIterator
from gitnetic.data.indexed_dataset import IndexedDatasetMixin
from gitnetic.optim import get_polynomial_decay_with_warmup, weight_decay_params
from gitnetic.utils import path_to_posix, perplexity

from .base import BaseTrainingMixin, TrainingParams

Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]


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
        dataset: IndexedDatasetMixin,
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


class TransformerDataModule(DataLoadingMixin, FromArgs, LightningDataModule):
    # pylint: disable=too-many-arguments
    def __init__(
        self,
        tokenizer: Tokenizer,
        train_data_prefix: Union[Text, Path],
        val_data_prefix: Union[Text, Path],
        max_tokens: Optional[int] = None,
        batch_size: Optional[int] = None,
        num_workers: int = 0,
    ) -> None:
        super().__init__(max_tokens, batch_size, num_workers)

        self.tokenizer = tokenizer
        self.train_data_prefix = train_data_prefix
        self.val_data_prefix = val_data_prefix

        self.train_dataset: Optional[IndexedDatasetMixin] = None
        self.train_iterator: Optional[Dataset] = None
        self.val_dataset: Optional[IndexedDatasetMixin] = None
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
        train_path = path_to_posix(self.train_data_prefix)
        self.train_dataset = IndexedDatasetMixin.from_file(train_path)
        self.train_iterator = self.get_dataset_itr(
            self.train_dataset, collator=self.collator, shuffle=True, drop_last=False
        )

        # prepare a validation dataset iterator
        val_path = path_to_posix(self.val_data_prefix)
        self.val_dataset = IndexedDatasetMixin.from_file(val_path)
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

    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # fmt: off
        parser.add_argument("--batch_size", type=int, default=None, required=False,
                            help="A number of instances/sentences in a batch.")
        parser.add_argument("--max_tokens", type=int, default=None, required=False,
                            help="A number of tokens in a batch.")
        parser.add_argument("--train_data_prefix", type=str, default=None, required=True,
                            help="A prefix path for the train dataset file.")
        parser.add_argument("--val_data_prefix", type=str, default=None, required=True,
                            help="A prefix path for the validation dataset file.")
        parser.add_argument("--num_workers", type=int, default=1, required=True,
                            help="A number of workers for data loading.")
        # fmt: on
        return parser


# pylint: disable=arguments-differ
class TransformerModule(BaseTrainingMixin, FromArgs, LightningModule):
    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: Tokenizer,
        training_params: TrainingParams,
    ) -> None:
        super().__init__(training_params)

        self.save_hyperparameters()

        self.model = model
        self.tokenizer = tokenizer

        # lazy initialized properties
        self.total_train_steps = 0
        self.lr_scheduler: Optional[LambdaLR] = None
        self._train_dataloader: Optional[DataLoader[Tensor]] = None
        self._val_dataloader: Optional[DataLoader[Tensor]] = None

        self.register_buffer("best_val_loss", torch.tensor(0.0))

    def setup(self, stage: Optional[Text] = None) -> None:
        del stage  # we don't use `stage` to build a module

        # calculate the total number of training steps
        assert self.trainer is not None
        if self.trainer.max_steps is not None:
            self.total_train_steps = self.trainer.max_steps
        else:
            datamodule = self.trainer.datamodule
            if isinstance(datamodule, TransformerDataModule):
                assert datamodule.train_iterator is not None
                train_batch_nums = len(datamodule.train_iterator)
                self.total_train_steps = self.training_steps(
                    train_batch_nums, self.trainer.max_epochs
                )
            else:
                raise ValueError(
                    "Unable to calculate the total steps."
                    " Please, include `--max_steps` argument"
                    " in your training command."
                )

    def forward(
        self, input_ids: Tensor, labels: Tensor, **kwargs: Any
    ) -> Tuple[Tensor, Tensor]:
        del kwargs  # we implement this method with our parameters
        self.model.train()
        outputs = self.model(input_ids=input_ids, labels=labels)
        loss, prediction_scores = outputs[:2]
        return loss, prediction_scores

    def prepare_batch(self, batch: Dict[Text, Tensor], batch_idx: int) -> None:
        del batch_idx  # nouse

        # data loader will produce an extra dimension
        # if a data iterator is used, so we have
        # to flatten our input tensor if this happens
        input_ids = batch["input_ids"]
        if len(input_ids.size()) == 3:
            batch["input_ids"] = input_ids.squeeze(0)

    def training_step(
        self, batch: Dict[Text, Tensor], batch_idx: int
    ) -> pl.TrainResult:
        self.prepare_batch(batch, batch_idx)
        loss, _ = self.forward(**batch)
        train_perplexity = perplexity(loss)
        batch_size = torch.tensor(len(batch["input_ids"]))

        # get the latest scheduled learning rate
        if self.lr_scheduler is None:
            learning_rate = torch.tensor(float("nan"))
        else:
            try:
                values = self.lr_scheduler.get_last_lr()  # type: ignore
                learning_rate = torch.tensor(values).mean()
            except IndexError:
                learning_rate = torch.tensor(float("nan"))

        result = pl.TrainResult(loss)
        result.log("train_loss", value=loss)
        result.log("train_ppl", value=train_perplexity)
        result.log("train_lr", value=learning_rate)
        result.log("train_bz", value=batch_size)

        # write training logs to the progress bar
        result.log_dict(
            {
                "ppl": train_perplexity,
                "lr": learning_rate,
                "bz": batch_size,
            },
            prog_bar=True,
            logger=False,
            on_step=True,
            on_epoch=False,
        )

        return result

    def validation_step(
        self, batch: Dict[Text, Tensor], batch_idx: int
    ) -> pl.EvalResult:
        self.prepare_batch(batch, batch_idx)
        loss, _ = self.forward(**batch)
        val_perplexity = perplexity(loss)
        result = pl.EvalResult()
        result.log("val_loss", value=loss, prog_bar=True)
        result.log("val_ppl", value=val_perplexity, prog_bar=True)
        return result

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[Dict]]:
        parameters = weight_decay_params(
            self.model,
            weight_decay=self.training_params.weight_decay,
            skip_list=["bias", "LayerNorm.weight"],
        )

        optimizer = AdamW(
            parameters,  # type: ignore
            betas=(0.9, 0.98),
            eps=1e-6,
            lr=self.training_params.learning_rate,
        )

        self.lr_scheduler = get_polynomial_decay_with_warmup(
            optimizer,
            num_warmup_steps=self.training_params.warmup_steps,
            num_training_steps=self.total_train_steps,
            power=self.training_params.power,
        )

        # called after each training steps
        step_scheduler = {
            "scheduler": self.lr_scheduler,
            "interval": "step",
        }

        return [optimizer], [step_scheduler]

    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # fmt: off
        parser.add_argument("--weight_decay", type=float, default=0.01, required=False,
                            help="A parameter for decaying weights while optimization steps.")
        parser.add_argument("--warmup_steps", type=int, default=4000, required=False,
                            help="A number of steps to get to the starting learning rate.")
        parser.add_argument("--learning_rate", type=float, default=5e-4, required=True,
                            help="A starting learning weight value after warmup.")
        parser.add_argument("--power", type=float, default=1.0, required=False,
                            help="A polynomial power for a learning rate scheduler.")
        # fmt: on
        return parser

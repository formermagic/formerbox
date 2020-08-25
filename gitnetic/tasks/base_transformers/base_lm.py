from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, List, Optional, Text, Tuple, Union

import torch
from pytorch_lightning import LightningDataModule
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    DataCollator,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from gitnetic.data.data_iterators import DatasetIterator
from gitnetic.data.indexed_dataset import IndexedDatasetMixin
from gitnetic.optim import get_polynomial_decay_with_warmup, weight_decay_params
from gitnetic.utils import path_to_posix, perplexity

from .base import BaseTrainingMixin, TrainingParams


class DataLoadingMixin:
    def __init__(
        self, max_tokens: Optional[int], batch_size: Optional[int], num_workers: int,
    ) -> None:
        super().__init__()
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.num_workers = num_workers

    def get_dataloader(
        self,
        dataset: IndexedDatasetMixin,
        collator: DataCollator,
        shuffle: bool,
        drop_last: bool,
    ) -> DataLoader:
        dataset_itr = DatasetIterator(
            dataset,
            collator=collator,
            max_tokens=self.max_tokens,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
        )

        return DataLoader(dataset_itr, num_workers=self.num_workers)


class TransformerDataModule(DataLoadingMixin, LightningDataModule):
    # pylint: disable=too-many-arguments
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        train_data_prefix: Union[Text, Path],
        val_data_prefix: Union[Text, Path],
        max_tokens: Optional[int],
        batch_size: Optional[int],
        num_workers: int,
    ) -> None:
        super().__init__(max_tokens, batch_size, num_workers)

        self.tokenizer = tokenizer
        self.train_data_prefix = train_data_prefix
        self.val_data_prefix = val_data_prefix

        self.train_dataset: Optional[IndexedDatasetMixin] = None
        self.val_dataset: Optional[IndexedDatasetMixin] = None

    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs  # no data to download

    def transfer_batch_to_device(self, batch: Any, device: torch.device) -> Any:
        del device  # lightning should have already moved a batch to the device
        return batch

    def setup(self, stage: Optional[Text] = None) -> None:
        del stage  # we don't use `stage` to build a dataloader

        # prepare a train dataloader
        train_path = path_to_posix(self.train_data_prefix)
        self.train_dataset = IndexedDatasetMixin.from_file(train_path)

        # prepare a validation dataloader
        val_path = path_to_posix(self.val_data_prefix)
        self.val_dataset = IndexedDatasetMixin.from_file(val_path)

    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        del args, kwargs  # use initialized properties to make a dataloader
        assert self.train_dataset is not None
        collator = DataCollatorForLanguageModeling(self.tokenizer)  # type: ignore
        return self.get_dataloader(
            self.train_dataset, collator=collator, shuffle=True, drop_last=False
        )

    def val_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        del args, kwargs  # use initialized properties to make a dataloader
        assert self.val_dataset is not None
        collator = DataCollatorForLanguageModeling(self.tokenizer)  # type: ignore
        return self.get_dataloader(
            self.val_dataset, collator=collator, shuffle=False, drop_last=False
        )

    def test_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        del args, kwargs  # use initialized properties to make a dataloader
        raise NotImplementedError()

    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # fmt: off
        parser.add_argument("--batch_size", type=int, default=None, required=False,
                            help="")
        parser.add_argument("--max_tokens", type=int, default=None, required=False,
                            help="")
        parser.add_argument("--train_data_prefix", type=str, default=None, required=True,
                            help="")
        parser.add_argument("--val_data_prefix", type=str, default=None, required=True,
                            help="")
        parser.add_argument("--num_workers", type=int, default=1, required=True,
                            help="")
        # fmt: on
        return parser


# pylint: disable=arguments-differ
class TransformerModule(BaseTrainingMixin):
    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        training_params: TrainingParams,
    ) -> None:
        super().__init__()

        self.save_hyperparameters()

        self.model = model
        self.tokenizer = tokenizer
        self.training_params = training_params

        # lazy initialized properties
        self.total_steps: Optional[
            int
        ] = None  # TODO: Figure out how to setup if max_epochs are given
        self.lr_scheduler: Optional[LambdaLR] = None
        self._train_dataloader: Optional[DataLoader[Tensor]] = None
        self._val_dataloader: Optional[DataLoader[Tensor]] = None

        self.register_buffer("best_val_loss", torch.tensor(0.0))

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
    ) -> Dict[Text, Union[Tensor, Dict[Text, Tensor]]]:
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

        return {
            "loss": loss,
            "log": {
                "train_loss": loss,
                "train_ppl": train_perplexity,
                "train_lr": learning_rate,
                "train_bz": batch_size,
            },
            "progress_bar": {
                "ppl": train_perplexity,
                "lr": learning_rate,
                "bz": batch_size,
            },
        }

    def validation_step(
        self, batch: Dict[Text, Tensor], batch_idx: int
    ) -> Dict[Text, Union[Tensor, Dict[Text, Tensor]]]:
        self.prepare_batch(batch, batch_idx)
        loss, _ = self.forward(**batch)
        val_perplexity = perplexity(loss)
        return {"val_loss": loss, "val_ppl": val_perplexity}

    def validation_epoch_end(
        self, outputs: List[Dict[Text, Tensor]]
    ) -> Dict[Text, Dict[Text, Tensor]]:
        avg_val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_val_ppl = torch.stack([x["val_ppl"] for x in outputs]).mean()
        return {
            "log": {"val_loss": avg_val_loss, "val_ppl": avg_val_ppl},
            "progress_bar": {"val_loss": avg_val_loss, "val_ppl": avg_val_ppl},
        }

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
            num_training_steps=self.total_steps or 0,
            power=self.training_params.power,
        )

        # called after each training steps
        step_scheduler = {
            "scheduler": self.lr_scheduler,
            "interval": "step",
        }

        return [optimizer], [step_scheduler]

    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser,) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # fmt: off
        parser.add_argument("--weight_decay", type=float, default=0.01, required=False,
                            help="")
        parser.add_argument("--warmup_steps", type=int, default=4000, required=False,
                            help="")
        parser.add_argument("--learning_rate", type=float, default=5e-4, required=True,
                            help="")
        parser.add_argument("--power", type=float, default=1.0, required=False,
                            help="")
        # fmt: on
        return parser

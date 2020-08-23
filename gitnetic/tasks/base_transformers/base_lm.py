from pathlib import Path
from typing import Any, Dict, List, Optional, Text, Tuple, Union

import torch
from pytorch_lightning import LightningDataModule
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import AdamW, DataCollator, PreTrainedModel, PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorForLanguageModeling

from gitnetic.data.indexed_dataset import IndexedDatasetMixin
from gitnetic.data.indexed_dataset_setup import IndexedDatasetSetup
from gitnetic.data.samplers import (
    BatchSampler,
    UniformBatchSampler,
    UniformMaxTokensBatchSampler,
)
from gitnetic.optim import get_polynomial_decay_with_warmup, weight_decay_params
from gitnetic.utils import path_to_posix, perplexity
from gitnetic.data.data_iterators import DatasetIterator
from .base import BaseTrainingMixin, DataParams, TrainingParams


class BaseDataModuleMixin:
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
        # prepare a batch sampler
        batch_sampler = self.batch_sampler(
            dataset=dataset,
            batch_size=self.batch_size,
            max_tokens=self.max_tokens,
            shuffle=shuffle,
            drop_last=drop_last,
        )

        return DataLoader(
            dataset,
            num_workers=self.num_workers,
            batch_sampler=batch_sampler,
            collate_fn=collator,
        )

    def batch_sampler(
        self,
        dataset: IndexedDatasetMixin,
        max_tokens: Optional[int],
        batch_size: Optional[int],
        shuffle: bool = True,
        drop_last: bool = False,
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


class BaseLMDataModule(BaseDataModuleMixin, LightningDataModule):
    # pylint: disable=too-many-arguments
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        train_data_prefix: Union[Text, Path],
        val_data_prefix: Union[Text, Path],
        dataset_impl: Text,
        max_tokens: Optional[int],
        batch_size: Optional[int],
        num_workers: int,
    ) -> None:
        super().__init__(max_tokens, batch_size, num_workers)

        self.tokenizer = tokenizer
        self.train_data_prefix = train_data_prefix
        self.val_data_prefix = val_data_prefix
        self.dataset_impl = dataset_impl

        # prepare the dataset type from the given impl
        self.dataset_setup = IndexedDatasetSetup.from_args(dataset_impl)

        self.train_dataset: Optional[IndexedDatasetMixin] = None
        self.val_dataset: Optional[IndexedDatasetMixin] = None

    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs  # no data to download

    def transfer_batch_to_device(self, batch: Any, device: torch.device) -> Any:
        pass

    def setup(self, stage: Optional[Text] = None) -> None:
        del stage  # we don't use `stage` to build a dataloader

        # prepare a dataset type class
        dataset_type = self.dataset_setup.dataset_type

        # prepare a train dataloader
        train_path = path_to_posix(self.train_data_prefix)
        self.train_dataset = dataset_type(filepath_prefix=train_path)

        # prepare a validation dataloader
        val_path = path_to_posix(self.val_data_prefix)
        self.val_dataset = dataset_type(filepath_prefix=val_path)

    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        del args, kwargs  # use initialized properties to make a dataloader
        assert self.train_dataset is not None
        collator = DataCollatorForLanguageModeling(self.tokenizer)  # type: ignore
        dataset = DatasetIterator(
            self.train_dataset,
            collator=collator,
            max_tokens=self.max_tokens,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )

        return DataLoader(dataset, num_workers=self.num_workers)

    def val_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        del args, kwargs  # use initialized properties to make a dataloader
        assert self.val_dataset is not None
        collator = DataCollatorForLanguageModeling(self.tokenizer)  # type: ignore
        dataset = DatasetIterator(
            self.val_dataset,
            collator=collator,
            max_tokens=self.max_tokens,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )

        return DataLoader(dataset, num_workers=self.num_workers)

    def test_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        del args, kwargs  # use initialized properties to make a dataloader
        raise NotImplementedError()


# pylint: disable=arguments-differ
class BaseLMTransformer(BaseTrainingMixin):
    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        training_params: TrainingParams,
        data_params: DataParams,
    ) -> None:
        super().__init__()

        self.save_hyperparameters()

        self.model = model
        self.tokenizer = tokenizer
        self.training_params = training_params
        self.data_params = data_params

        # lazy initialized properties
        self.total_steps: Optional[int] = None
        self.lr_scheduler: Optional[LambdaLR] = None
        self._train_dataloader: Optional[DataLoader[Tensor]] = None
        self._val_dataloader: Optional[DataLoader[Tensor]] = None

    def forward(
        self, input_ids: torch.Tensor, labels: torch.Tensor, **kwargs: Any
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
        batch_size = torch.tensor([len(batch["input_ids"])])

        # get the latest scheduled learning rate
        if self.lr_scheduler is None:
            learning_rate = torch.tensor([float("nan")])
        else:
            try:
                values = self.lr_scheduler.get_last_lr()  # type: ignore
                learning_rate = torch.tensor(values).mean()
            except IndexError:
                learning_rate = torch.tensor([float("nan")])

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

    # def prepare_data(self) -> None:
    #     pass

    # def setup(self, stage: Text) -> None:
    #     del stage  # we don't use `stage` to build a dataloader

    #     # prepare the dataset type from the given impl
    #     dataset_setup = IndexedDatasetSetup.from_args(self.data_params.dataset_impl)
    #     dataset_type = dataset_setup.dataset_type

    #     # prepare a language modeling collator
    #     collator = DataCollatorForLanguageModeling(self.tokenizer)  # type: ignore

    #     # prepare a train dataloader
    #     train_path = path_to_posix(self.data_params.train_data_prefix)
    #     train_dataset = dataset_type(filepath_prefix=train_path)
    #     self._train_dataloader = self.get_dataloader(
    #         train_dataset, collator=collator, shuffle=True, drop_last=False,
    #     )

    #     # prepare a validation dataloader
    #     val_path = path_to_posix(self.data_params.val_data_prefix)
    #     val_dataset = dataset_type(filepath_prefix=val_path)
    #     self._val_dataloader = self.get_dataloader(
    #         val_dataset, collator=collator, shuffle=False, drop_last=False,
    #     )

    #     # calculate the total training steps
    #     if getattr(self.trainer, "max_steps") is None:
    #         self.total_steps = self.training_steps(len(self._train_dataloader))
    #     else:
    #         self.total_steps = self.trainer.max_steps

    # def train_dataloader(self) -> DataLoader:
    #     assert self._train_dataloader is not None
    #     return self._train_dataloader

    # def val_dataloader(self) -> DataLoader:
    #     assert self._val_dataloader is not None
    #     return self._val_dataloader

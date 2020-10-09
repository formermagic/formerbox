from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Text, Tuple, Type, Union

import torch
from gitnetic.common.dataclass_argparse import DataclassBase
from gitnetic.common.has_params import HasParsableParams
from gitnetic.common.registrable import Registrable
from gitnetic.data.dataset_iterators import DatasetIterator
from gitnetic.data.indexed_dataset import IndexedDatasetBase
from gitnetic.optim import AdamW, get_polynomial_decay_with_warmup, weight_decay_params
from gitnetic.utils import path_to_posix, perplexity
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.core.datamodule import _DataModuleWrapper
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from transformers import (
    DataCollator,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from typing_extensions import Protocol, _ProtocolMeta  # type: ignore

from .base import BaseTrainingMixin

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


class TransformerModuleOutput(Protocol):
    loss: Optional[torch.FloatTensor]
    logits: torch.FloatTensor
    hidden_states: Optional[Tuple[torch.FloatTensor]]
    attentions: Optional[Tuple[torch.FloatTensor]]


# pylint: disable=arguments-differ
class TransformerModule(
    BaseTrainingMixin,
    LightningModule,
    Registrable,
    HasParsableParams,
):
    @dataclass
    class Params(DataclassBase):
        weight_decay: float = field(
            default=0.01,
            metadata={
                "help": "A parameter for decaying weights while optimization steps."
            },
        )
        warmup_steps: int = field(
            default=4000,
            metadata={
                "help": "A number of steps to get to the starting learning rate."
            },
        )
        learning_rate: float = field(
            default=5e-4,
            metadata={"help": "A starting learning weight value after warmup."},
        )
        learning_rate_end: float = field(
            default=1e-5,
            metadata={"help": "The learning rate value after last training step."},
        )
        power: float = field(
            default=1.0,
            metadata={"help": "A polynomial power for a learning rate scheduler."},
        )

    params: Params
    params_type: Type[Params] = Params

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: Tokenizer,
        params: Params,
    ) -> None:
        super().__init__()

        # save the arguments to easily restore
        # from the saved pytorch checkpoint
        self.save_hyperparameters()

        self.model = model
        self.tokenizer = tokenizer
        self.params = params

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
        self,
        input_ids: Tensor,
        labels: Tensor,
        return_dict: bool = True,
        **kwargs: Any,
    ) -> TransformerModuleOutput:
        # put the module into train mode
        self.model.train()

        # prepare model forward pass arguments
        kwargs.setdefault("input_ids", input_ids)
        kwargs.setdefault("labels", labels)
        kwargs.setdefault("return_dict", return_dict)

        # make a forward pass with our transformer model
        outputs = self.model(**kwargs)
        # the language model should return a `TransformerModuleOutput` instance
        assert isinstance(outputs, TransformerModuleOutput)

        # return the model outputs
        return outputs

    def prepare_batch(self, batch: Dict[Text, Tensor], batch_idx: int) -> None:
        del batch_idx  # nouse

        # data loader will produce an extra dimension
        # if a data iterator is used, so we have
        # to flatten our input tensor if this happens
        input_ids = batch["input_ids"]
        if len(input_ids.size()) == 3:
            batch["input_ids"] = input_ids.squeeze(0)

    def training_step(
        self,
        batch: Dict[Text, Tensor],
        batch_idx: int,
        optimizer_idx: Optional[int] = None,
        hiddens: Optional[Tensor] = None,
    ) -> Dict[Text, Any]:
        del optimizer_idx, hiddens  # nouse
        self.prepare_batch(batch, batch_idx)

        # model forward pass & prepare metrics values
        outputs = self.forward(**batch)
        assert outputs.loss is not None
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

        # log training metrics
        self.log("train_loss", outputs.loss, prog_bar=True)
        self.log("train_lr", learning_rate, prog_bar=True)
        self.log("train_bsz", batch_size, prog_bar=True)

        return {
            "loss": outputs.loss,
            "lr": learning_rate,
            "bsz": batch_size,
        }

    def validation_step(
        self, batch: Dict[Text, Tensor], batch_idx: int, **kwargs: Any
    ) -> None:
        del kwargs  # nouse
        self.prepare_batch(batch, batch_idx)
        # model forward pass & prepare metrics
        outputs = self.forward(**batch)
        assert outputs.loss is not None
        # log validation metrics
        self.log("val_loss", outputs.loss, prog_bar=True)

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[Dict]]:
        parameters = weight_decay_params(
            self.model,
            weight_decay=self.params.weight_decay,
            skip_list=["bias", "LayerNorm.weight"],
        )

        optimizer = AdamW(
            parameters,
            betas=(0.9, 0.98),
            eps=1e-6,
            lr=self.params.learning_rate,
        )

        self.lr_scheduler = get_polynomial_decay_with_warmup(
            optimizer,
            num_warmup_steps=self.params.warmup_steps,
            num_training_steps=self.total_train_steps,
            learning_rate_end=self.params.learning_rate_end,
            power=self.params.power,
        )

        # called after each training steps
        step_scheduler = {
            "scheduler": self.lr_scheduler,
            "interval": "step",
        }

        return [optimizer], [step_scheduler]

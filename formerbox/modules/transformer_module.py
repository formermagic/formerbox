import logging
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Text, Tuple, Type

import torch
from formerbox.common.dataclass_argparse import DataclassBase
from formerbox.common.has_params import HasParsableParams
from formerbox.common.registrable import Registrable
from formerbox.modules.lightning_module_mixin import LightningModuleMixin
from formerbox.modules.metrics import Perplexity
from formerbox.modules.transformer_datamodule import TransformerDataModule
from formerbox.optim import AdamW, get_polynomial_decay_with_warmup, weight_decay_params
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizerFast as Tokenizer
from typing_extensions import Protocol

logger = logging.getLogger(__name__)


class TransformerModuleOutput(Protocol):
    loss: Optional[torch.FloatTensor]
    logits: torch.FloatTensor
    hidden_states: Optional[Tuple[torch.FloatTensor]]
    attentions: Optional[Tuple[torch.FloatTensor]]


# pylint: disable=arguments-differ
class TransformerModule(
    LightningModuleMixin,
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

        # metrics properties
        self.perplexity = Perplexity()

        # lazy initialized properties
        self.total_train_steps = 0
        self.lr_scheduler: Optional[LambdaLR] = None

    @property
    def learning_rate(self) -> torch.Tensor:
        learning_rate: torch.Tensor
        if self.lr_scheduler is None:
            learning_rate = torch.tensor(float("nan"))
        else:
            try:
                values = self.lr_scheduler.get_last_lr()  # type: ignore
                learning_rate = torch.tensor(values).mean()
            except IndexError:
                learning_rate = torch.tensor(float("nan"))

        return learning_rate

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

    @abstractmethod
    def forward(
        self,
        input_ids: Tensor,
        labels: Tensor,
        return_dict: bool = True,
        **kwargs: Any,
    ) -> TransformerModuleOutput:
        raise NotImplementedError()

    @abstractmethod
    def training_step(
        self,
        batch: Dict[Text, Tensor],
        batch_idx: int,
        optimizer_idx: Optional[int] = None,
        hiddens: Optional[Tensor] = None,
    ) -> Dict[Text, Any]:
        raise NotImplementedError()

    @abstractmethod
    def validation_step(
        self,
        batch: Dict[Text, Tensor],
        batch_idx: int,
        **kwargs: Any,
    ) -> None:
        raise NotImplementedError()

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

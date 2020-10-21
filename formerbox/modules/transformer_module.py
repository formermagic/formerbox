import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Text, Tuple, Type, Union

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
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from typing_extensions import Protocol

Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

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

        self.perplexity = Perplexity(tokenizer.vocab_size)

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

        # prepare detached tensors for logging
        loss = outputs.loss.detach().cpu()
        logits = outputs.logits.detach()
        labels = batch["labels"].detach()

        perplexity = self.perplexity(logits, labels)
        perplexity = perplexity.detach().cpu()
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
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_ppl", perplexity, prog_bar=True)
        self.log("train_lr", learning_rate, prog_bar=True)
        self.log("train_bsz", batch_size, prog_bar=True)

        return {
            "loss": outputs.loss,
            "ppl": perplexity,
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

        # prepare detached tensors for logging
        loss = outputs.loss.detach().cpu()
        logits = outputs.logits.detach()
        labels = batch["labels"].detach()

        perplexity = self.perplexity(logits, labels)
        perplexity = perplexity.detach().cpu()

        # log validation metrics
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_ppl", perplexity, prog_bar=True)

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

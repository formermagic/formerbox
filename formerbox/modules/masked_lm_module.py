import logging
import typing
from dataclasses import dataclass
from typing import Any, Dict, Optional, Text, Type

import torch
from formerbox.modules.transformer_module import TransformerModule
from torch import Tensor
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizerFast as Tokenizer
from transformers.file_utils import ModelOutput

logger = logging.getLogger(__name__)


# pylint: disable=arguments-differ
class MaskedLMModule(TransformerModule):
    @dataclass
    class Params(TransformerModule.Params):
        pass

    params: Params
    params_type: Type[Params] = Params

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: Tokenizer,
        params: Params,
    ) -> None:
        super().__init__(model, tokenizer, params)

        # save the arguments to easily restore
        # from the saved pytorch checkpoint
        self.save_hyperparameters()

    def forward(
        self,
        input_ids: Tensor,
        labels: Tensor,
        return_dict: bool = True,
        **kwargs: Any,
    ) -> ModelOutput:
        # put the module into train mode
        self.model.train()

        # prepare model forward pass arguments
        kwargs.setdefault("input_ids", input_ids)
        kwargs.setdefault("labels", labels)
        kwargs.setdefault("return_dict", return_dict)

        # make a forward pass with our transformer model
        outputs = self.model.forward(**kwargs)
        # the model should return a `ModelOutput` instance
        assert isinstance(outputs, ModelOutput)

        # return the model outputs
        return outputs

    def training_step(
        self,
        batch: Dict[Text, Tensor],
        batch_idx: int,
        optimizer_idx: Optional[int] = None,
        hiddens: Optional[Tensor] = None,
    ) -> Tensor:
        del batch_idx, optimizer_idx, hiddens  # nouse

        # make a model forward pass
        model_output = self.forward(**batch)

        # get the loss based on model output
        loss = model_output["loss"]
        loss = typing.cast(Tensor, loss)

        # prepare other metrics to log
        perplexity = self.perplexity.forward(loss.detach())
        batch_size = torch.tensor(batch["input_ids"].size(0))
        learning_rate = self.learning_rate

        metrics = {
            "train_loss": loss,
            "train_ppl": perplexity,
            "train_lr": learning_rate,
            "train_bsz": batch_size,
            "global_step": self.trainer.global_step,
        }

        # log training metrics
        self.log_dict(metrics, prog_bar=True)
        # this ensures that we'll always log the last step
        self.log("step", self.trainer.global_step)

        return loss

    def validation_step(
        self,
        batch: Dict[Text, Tensor],
        batch_idx: int,
        **kwargs: Any,
    ) -> None:
        del batch_idx, kwargs  # nouse

        # make a model forward pass
        model_output = self.forward(**batch)

        # get the loss based on model output
        loss = model_output["loss"]
        loss = typing.cast(Tensor, loss)

        # prepare other metrics to log
        perplexity = self.perplexity.forward(loss.detach())
        metrics = {"val_loss": loss, "val_ppl": perplexity}

        # log validation metrics
        self.log_dict(metrics, prog_bar=True)

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Text, Type

import torch
import torch.nn as nn
from formerbox.modules.transformer_module import TransformerModule
from formerbox.modules.utils import LabelSmoothingNLLLoss
from torch import Tensor
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizerFast as Tokenizer
from typing_extensions import Protocol

logger = logging.getLogger(__name__)


class Seq2SeqModuleOutput(Protocol):
    logits: torch.Tensor


# pylint: disable=arguments-differ
class Seq2SeqModule(TransformerModule):
    @dataclass
    class Params(TransformerModule.Params):
        label_smoothing: float = field(
            default=0.0,
            metadata={"help": ""},
        )

    params: Params
    params_type: Type[Params] = Params
    criterion: nn.Module

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

        assert self.tokenizer.pad_token_id is not None
        self.criterion = LabelSmoothingNLLLoss(
            label_smoothing=self.params.label_smoothing,
            ignore_index=self.tokenizer.pad_token_id,
        )

    def forward(
        self,
        input_ids: Tensor,
        labels: Tensor,
        attention_mask: Tensor,
        return_dict: bool = True,
        **kwargs: Any,
    ) -> Seq2SeqModuleOutput:
        # put the module into train mode
        self.model.train()

        # prepare model forward pass arguments
        kwargs.setdefault("input_ids", input_ids)
        kwargs.setdefault("labels", labels)
        kwargs.setdefault("attention_mask", attention_mask)
        kwargs.setdefault("return_dict", return_dict)

        # make a forward pass with our transformer model
        outputs = self.model(**kwargs)
        # the model should return a `ModelOutput` instance
        assert isinstance(outputs, Seq2SeqModuleOutput)

        # return the model outputs
        return outputs

    def training_step(
        self,
        batch: Dict[Text, Tensor],
        batch_idx: int,
        optimizer_idx: Optional[int] = None,
        hiddens: Optional[Tensor] = None,
    ) -> Dict[Text, Any]:
        del batch_idx, optimizer_idx, hiddens  # nouse

        # make a model forward pass
        outputs = self.forward(**batch)

        # compute the loss based on model's outputs
        logits = outputs.logits
        targets = batch["labels"]
        loss = self.criterion(logits, targets)
        assert isinstance(loss, torch.Tensor)

        # prepare other metrics to log
        perplexity = self.perplexity(loss.detach())
        batch_size = torch.tensor(batch["input_ids"].size(0))
        learning_rate = self.learning_rate

        # log training metrics
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_ppl", perplexity, prog_bar=True)
        self.log("train_lr", learning_rate, prog_bar=True)
        self.log("train_bsz", batch_size, prog_bar=True)

        return {
            "loss": loss,
            "ppl": perplexity,
            "lr": learning_rate,
            "bsz": batch_size,
        }

    def validation_step(
        self,
        batch: Dict[Text, Tensor],
        batch_idx: int,
        **kwargs: Any,
    ) -> None:
        del batch_idx, kwargs  # nouse

        # make a model forward pass
        outputs = self.forward(**batch)

        # compute the loss based on model's outputs
        logits = outputs.logits
        targets = batch["labels"]
        loss = self.criterion(logits, targets)
        assert isinstance(loss, Tensor)

        # prepare other metrics to log
        perplexity = self.perplexity(loss.detach())
        perplexity = perplexity.detach().cpu()

        # log validation metrics
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_ppl", perplexity, prog_bar=True)

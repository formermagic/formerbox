import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Text, Tuple, Type

import torch
from formerbox.modules.transformer_module import TransformerModule
from formerbox.modules.utils import LabelSmoothingNLLLoss
from torch import Tensor
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizerFast as Tokenizer
from transformers.file_utils import ModelOutput

logger = logging.getLogger(__name__)


# pylint: disable=arguments-differ
class TranslationModule(TransformerModule):
    @dataclass
    class Params(TransformerModule.Params):
        label_smoothing: float = field(
            default=0.0,
            metadata={
                "help": "A value for calculating the label smoothed loss."
                " Setting `label_smoothing=0` gives a regular cross entropy loss."
                " Default value is `label_smoothing=0`."
            },
        )

    params: Params
    params_type: Type[Params] = Params
    criterion: LabelSmoothingNLLLoss

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
    ) -> ModelOutput:
        # put the module into train mode
        self.model.train()

        # prepare model forward pass arguments
        kwargs.setdefault("input_ids", input_ids)
        kwargs.setdefault("labels", labels)
        kwargs.setdefault("attention_mask", attention_mask)
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

        # compute the loss based on model output
        loss, nll_loss = self._calculate_loss(batch, model_output)

        # prepare other metrics to log
        perplexity = self.perplexity.forward(loss.detach())
        batch_size = torch.tensor(batch["input_ids"].size(0))
        learning_rate = self.learning_rate

        metrics = {
            "train_loss": loss,
            "train_nll_loss": nll_loss,
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

        # compute the loss based on model output
        loss, nll_loss = self._calculate_loss(batch, model_output)

        # prepare other metrics to log
        perplexity = self.perplexity.forward(loss.detach())
        metrics = {
            "val_loss": loss,
            "val_nll_loss": nll_loss,
            "val_ppl": perplexity,
        }

        # log validation metrics
        self.log_dict(metrics, prog_bar=True)

    def _calculate_loss(
        self, batch: Dict[Text, Tensor], model_output: ModelOutput
    ) -> Tuple[Tensor, Tensor]:
        # get input target values
        targets: Tensor = batch["labels"]
        # get model output logits
        logits: Tensor = model_output["logits"]
        # get model output nll loss
        nll_loss: Tensor = model_output["loss"]

        # calculate label smoothed loss
        smoothed_loss: Tensor = self.criterion.forward(logits, targets)

        return smoothed_loss, nll_loss

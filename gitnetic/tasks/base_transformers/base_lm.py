from typing import Any, Dict, List, Optional, Text, Tuple, Union

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import AdamW, PreTrainedModel, PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorForLanguageModeling

from gitnetic.data.indexed_dataset_setup import IndexedDatasetSetup
from gitnetic.optim import get_polynomial_decay_with_warmup, weight_decay_params
from gitnetic.utils import path_to_posix, perplexity

from .base import BaseTrainingMixin, DataParams, TrainingParams


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
        self, input_ids: torch.LongTensor, labels: torch.LongTensor, **kwargs: Any
    ) -> Tuple[Tensor, Tensor]:
        del kwargs  # we implement this method with our parameters
        self.model.train()
        outputs = self.model(input_ids=input_ids, labels=labels)
        loss, prediction_scores = outputs[:2]
        return loss, prediction_scores

    def training_step(
        self, batch: Dict[Text, Tensor], batch_idx: int
    ) -> Dict[Text, Union[Tensor, Dict[Text, Tensor]]]:
        del batch_idx  # we don't use `batch_idx` now
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
        del batch_idx  # we don't use `batch_idx` now
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

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Text) -> None:
        del stage  # we don't use `stage` to build a dataloader

        # prepare the dataset type from the given impl
        dataset_setup = IndexedDatasetSetup.from_args(self.data_params.dataset_impl)
        dataset_type = dataset_setup.dataset_type

        # prepare a language modeling collator
        collator = DataCollatorForLanguageModeling(self.tokenizer)  # type: ignore

        # prepare a train dataloader
        train_path = path_to_posix(self.data_params.train_data_prefix)
        train_dataset = dataset_type(filepath_prefix=train_path)
        self._train_dataloader = self.get_dataloader(
            train_dataset, collator=collator, shuffle=True, drop_last=False,
        )

        # prepare a validation dataloader
        val_path = path_to_posix(self.data_params.val_data_prefix)
        val_dataset = dataset_type(filepath_prefix=val_path)
        self._val_dataloader = self.get_dataloader(
            val_dataset, collator=collator, shuffle=False, drop_last=False,
        )

        # calculate the total training steps
        if getattr(self.trainer, "max_steps") is None:
            self.total_steps = self.training_steps(len(self._train_dataloader))
        else:
            self.total_steps = self.trainer.max_steps

    def train_dataloader(self) -> DataLoader:
        assert self._train_dataloader is not None
        return self._train_dataloader

    def val_dataloader(self) -> DataLoader:
        assert self._val_dataloader is not None
        return self._val_dataloader

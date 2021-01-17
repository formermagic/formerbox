import logging
from dataclasses import dataclass
from typing import Optional, Text, Type

from formerbox.data.indexed_dataset import IndexedDatasetBase
from formerbox.modules.transformer_datamodule import TransformerDataModule
from transformers import DataCollatorForLanguageModeling
from transformers import PreTrainedTokenizerFast as Tokenizer

logger = logging.getLogger(__name__)


class MaskedLMDataModule(TransformerDataModule):
    @dataclass
    class Params(TransformerDataModule.Params):
        pass

    params: Params
    params_type: Type[Params] = Params

    def __init__(self, tokenizer: Tokenizer, params: Params) -> None:
        super().__init__(tokenizer, params)

        self.collator = DataCollatorForLanguageModeling(self.tokenizer)

    def setup(self, stage: Optional[Text] = None) -> None:
        del stage  # we don't use `stage` to build a dataloader

        # prepare a train dataset iterator
        train_path = str(self.params.train_data_prefix)
        self.train_dataset = IndexedDatasetBase.from_file(train_path)
        self.train_iterator = self.get_dataset_itr(
            self.train_dataset,
            collator=self.collator,
            shuffle=True,
            drop_last=False,
        )

        # prepare a validation dataset iterator
        val_path = str(self.params.val_data_prefix)
        self.val_dataset = IndexedDatasetBase.from_file(val_path)
        self.val_iterator = self.get_dataset_itr(
            self.val_dataset,
            collator=self.collator,
            shuffle=False,
            drop_last=False,
        )

import logging
from dataclasses import dataclass, field
from typing import Optional, Text, Type

from formerbox.common.dataclass_argparse import MISSING
from formerbox.data.data_collator import DataCollatorForTranslation
from formerbox.data.translation_dataset import TranslationDataset
from formerbox.modules.transformer_datamodule import TransformerDataModule
from transformers import PreTrainedTokenizerFast as Tokenizer

logger = logging.getLogger(__name__)


class TranslationDataModule(TransformerDataModule):
    @dataclass
    class Params(TransformerDataModule.Params):
        src_lang: Text = field(
            default=MISSING,
            metadata={
                "help": "A source language extension."
                " Used in mono-bilingual translation tasks."
            },
        )
        tgt_lang: Optional[Text] = field(
            default=None,
            metadata={
                "help": "A target language extension."
                " Used in bilingual translation tasks."
                " Default value is `None`."
            },
        )

    params: Params
    params_type: Type[Params] = Params

    def __init__(self, tokenizer: Tokenizer, params: Params) -> None:
        super().__init__(tokenizer, params)

        self.collator = DataCollatorForTranslation(self.tokenizer)

    def setup(self, stage: Optional[Text] = None) -> None:
        del stage  # we don't use `stage` to build a dataloader

        # prepare a train dataset iterator
        train_path = str(self.params.train_data_prefix)
        self.train_dataset = TranslationDataset.from_file(
            train_path,
            src_lang=self.params.src_lang,
            tgt_lang=self.params.tgt_lang,
        )
        self.train_iterator = self.get_dataset_itr(
            self.train_dataset,
            collator=self.collator,
            shuffle=True,
            drop_last=False,
        )

        # prepare a validation dataset iterator
        val_path = str(self.params.val_data_prefix)
        self.val_dataset = TranslationDataset.from_file(
            val_path,
            src_lang=self.params.src_lang,
            tgt_lang=self.params.tgt_lang,
        )
        self.val_iterator = self.get_dataset_itr(
            self.val_dataset,
            collator=self.collator,
            shuffle=False,
            drop_last=False,
        )

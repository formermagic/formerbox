import logging
from dataclasses import dataclass, field
from typing import Optional, Text, Type

from formerbox.common.dataclass_argparse import MISSING
from formerbox.data.data_collator import DataCollatorForWholeWordMasking, ReplaceLength
from formerbox.data.translation_dataset import TranslationDataset
from formerbox.modules.transformer_datamodule import TransformerDataModule
from transformers import PreTrainedTokenizerFast as Tokenizer

logger = logging.getLogger(__name__)


class WordLMDataModule(TransformerDataModule):
    @dataclass
    class Params(TransformerDataModule.Params):
        lang: Text = field(
            default=MISSING,
            metadata={
                "help": "A language extension of the dataset files.",
            },
        )
        masked_token_ratio: float = field(
            default=0.15,
            metadata={
                "help": "A ratio of input tokens to mask.",
            },
        )
        random_token_ratio: float = field(
            default=0.0,
            metadata={
                "help": "A ratio of random tokens among ones selected for masking."
            },
        )
        replace_length: ReplaceLength = field(
            default=-1,
            metadata={
                "help": "A masked replacement length for whole word masking."
                " `replace_length=0` removes masked subwords from input,"
                " `replace_length=1` replaces all masked subwords with one token,"
                " `replace_length=-1` replaces each masked subwords with mask token."
                " Default value is `replace_length=-1`."
            },
        )

    params: Params
    params_type: Type[Params] = Params

    def __init__(self, tokenizer: Tokenizer, params: Params) -> None:
        super().__init__(tokenizer, params)

        self.collator = DataCollatorForWholeWordMasking(
            self.tokenizer,
            masked_token_ratio=params.masked_token_ratio,
            random_token_ratio=params.random_token_ratio,
            replace_length=params.replace_length,
        )

    def setup(self, stage: Optional[Text] = None) -> None:
        del stage  # we don't use `stage` to build a dataloader

        # prepare a train dataset iterator
        train_path = str(self.params.train_data_prefix)
        self.train_dataset = TranslationDataset.from_file(
            train_path, src_lang=self.params.lang
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
            val_path, src_lang=self.params.lang
        )
        self.val_iterator = self.get_dataset_itr(
            self.val_dataset,
            collator=self.collator,
            shuffle=False,
            drop_last=False,
        )

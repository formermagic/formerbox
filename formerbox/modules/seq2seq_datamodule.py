import logging
from dataclasses import dataclass, field
from typing import Optional, Text, Type

from formerbox.common.dataclass_argparse import MISSING
from formerbox.data.data_collator import DataCollatorForSeq2SeqDenoising, ReplaceLength
from formerbox.data.seq2seq_dataset import Seq2SeqDataset
from formerbox.modules.transformer_datamodule import TransformerDataModule
from transformers import PreTrainedTokenizerFast as Tokenizer

logger = logging.getLogger(__name__)


class Seq2SeqDataModule(TransformerDataModule):
    @dataclass
    class Params(TransformerDataModule.Params):
        src_lang: Text = field(
            default=MISSING,
            metadata={"help": ""},
        )
        tgt_lang: Optional[Text] = field(
            default=None,
            metadata={"help": ""},
        )
        masked_token_ratio: float = field(
            default=0.15,
            metadata={"help": ""},
        )
        random_token_ratio: float = field(
            default=0.0,
            metadata={"help": ""},
        )
        replace_length: ReplaceLength = field(
            default=-1,
            metadata={"help": ""},
        )

    params: Params
    params_type: Type[Params] = Params

    def __init__(self, tokenizer: Tokenizer, params: Params) -> None:
        super().__init__(tokenizer, params)

        self.collator = DataCollatorForSeq2SeqDenoising(
            self.tokenizer,
            masked_token_ratio=params.masked_token_ratio,
            random_token_ratio=params.random_token_ratio,
            replace_length=params.replace_length,
        )

    def setup(self, stage: Optional[Text] = None) -> None:
        del stage  # we don't use `stage` to build a dataloader

        # prepare a train dataset iterator
        train_path = str(self.params.train_data_prefix)
        self.train_dataset = Seq2SeqDataset.from_file(
            train_path, src_lang=self.params.src_lang, tgt_lang=self.params.tgt_lang
        )
        self.train_iterator = self.get_dataset_itr(
            self.train_dataset, collator=self.collator, shuffle=True, drop_last=False
        )

        # prepare a validation dataset iterator
        val_path = str(self.params.val_data_prefix)
        self.val_dataset = Seq2SeqDataset.from_file(
            val_path, src_lang=self.params.src_lang, tgt_lang=self.params.tgt_lang
        )
        self.val_iterator = self.get_dataset_itr(
            self.val_dataset, collator=self.collator, shuffle=False, drop_last=False
        )

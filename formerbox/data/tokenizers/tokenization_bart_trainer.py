import logging
from pathlib import Path
from typing import Any, Optional, Text, Union

from formerbox.modules import TokenizerTrainer
from formerbox.data.tokenizers.tokenization_bart import BartTokenizer
from formerbox.data.tokenizers.tokenization_roberta_trainer import (
    RobertaTokenizerTrainer,
)

logger = logging.getLogger(__name__)


@TokenizerTrainer.register(name="bart", constructor="from_partial")
class BartTokenizerTrainer(RobertaTokenizerTrainer):
    class Params(RobertaTokenizerTrainer.Params):
        pass

    def configure_tokenizer(
        self, tokenizer_path: Union[Text, Path], **kwargs: Any
    ) -> BartTokenizer:
        # prepare paths to the tokenizer files
        if isinstance(tokenizer_path, str):
            tokenizer_path = Path(tokenizer_path)
        vocab_file = str(tokenizer_path / "vocab.json")
        merges_file = str(tokenizer_path / "merges.txt")

        # prepare the unified pre-trained tokenizer path
        # tokenizers will produce this file if no legacy
        # format is specified while saving
        tokenizer_file: Optional[Text] = None
        if not self.params.legacy_format:
            tokenizer_file = str(tokenizer_path / "tokenizer.json")

        # merge user-defined arguments into kwargs
        kwargs.update(self.get_tokenizer_args(self.params))

        # configure the pretrained tokenizer
        return BartTokenizer(
            vocab_file=vocab_file,
            merges_file=merges_file,
            tokenizer_file=tokenizer_file,
            **kwargs,
        )

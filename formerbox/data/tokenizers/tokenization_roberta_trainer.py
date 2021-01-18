import logging
from pathlib import Path
from typing import Any, Optional, Text, Union

from formerbox.data.tokenizers.tokenization_gpt2_trainer import GPT2TokenizerTrainer
from formerbox.data.tokenizers.tokenization_roberta import RobertaTokenizer
from formerbox.modules import TokenizerTrainer

logger = logging.getLogger(__name__)

# pylint: disable=arguments-differ
@TokenizerTrainer.register(name="roberta", constructor="from_partial")
class RobertaTokenizerTrainer(GPT2TokenizerTrainer):
    class Params(GPT2TokenizerTrainer.Params):
        pass

    def configure_tokenizer(
        self, tokenizer_path: Union[Text, Path], **kwargs: Any
    ) -> RobertaTokenizer:
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
        return RobertaTokenizer(
            vocab_file=vocab_file,
            merges_file=merges_file,
            tokenizer_file=tokenizer_file,
            **kwargs,
        )

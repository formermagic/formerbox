import logging
from pathlib import Path
from typing import Any, Optional, Text, Union

from formerbox.data.tokenizers import RobertaTokenizerTrainer
from formerbox.modules import TokenizerTrainer
from formerbox.tasks.code.tokenization_code_roberta import CodeRobertaTokenizer
from formerbox.utils.code_tokenizer import SpecialToken

logger = logging.getLogger(__name__)


@TokenizerTrainer.register("code_roberta", constructor="from_partial")
class CodeRobertaTokenizerTrainer(RobertaTokenizerTrainer):
    Params = RobertaTokenizerTrainer.Params

    def __init__(self, params: Params, **kwargs: Any) -> None:
        super().__init__(params, **kwargs)

        # add code special tokens as additional tokens
        self.additional_tokens = [token.value for token in SpecialToken]

    def configure_tokenizer(
        self, tokenizer_path: Union[Text, Path], **kwargs: Any
    ) -> CodeRobertaTokenizer:
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
        return CodeRobertaTokenizer(
            vocab_file=vocab_file,
            merges_file=merges_file,
            tokenizer_file=tokenizer_file,
            **kwargs,
        )

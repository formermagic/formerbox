import logging
from pathlib import Path
from typing import Any, Optional, Text, Union

from formerbox.modules import TokenizerModule
from formerbox.tasks import ByteLevelBPETokenizerModule
from formerbox.tasks.code.code_tokenization import CodeBBPETokenizerFast
from formerbox.utils.code_tokenizer import SpecialToken
from tokenizers import AddedToken

Token = Union[Text, AddedToken]

logger = logging.getLogger(__name__)


@TokenizerModule.register("code-bbpe-tokenizer", constructor="from_partial")
class CodeBBPETokenizerModule(ByteLevelBPETokenizerModule):
    Params = ByteLevelBPETokenizerModule.Params

    def __init__(self, params: Params, **kwargs: Any) -> None:
        super().__init__(params, **kwargs)

        # add code special tokens
        for token in SpecialToken:
            self.special_tokens.append(token.value)

    def configure_tokenizer(
        self, tokenizer_path: Union[Text, Path], **kwargs: Any
    ) -> CodeBBPETokenizerFast:
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
        return CodeBBPETokenizerFast(
            vocab_file=vocab_file,
            merges_file=merges_file,
            tokenizer_file=tokenizer_file,
            **kwargs,
        )

    @classmethod
    def from_pretrained(
        cls, tokenizer_path: Union[Text, Path], **kwargs: Any
    ) -> CodeBBPETokenizerFast:
        # convert `tokenizer_path` to string
        if isinstance(tokenizer_path, Path):
            tokenizer_path = str(tokenizer_path)

        # load the pretrained tokenizer
        tokenizer = CodeBBPETokenizerFast.from_pretrained(tokenizer_path, **kwargs)

        return tokenizer

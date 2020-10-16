from pathlib import Path
from typing import Any, Text, Union

from formerbox.modules import TokenizerModule
from formerbox.tasks import TransformerTokenizerModule
from formerbox.tasks.code.code_tokenization import CodeTokenizerFast
from formerbox.utils.code_tokenizer import SpecialToken
from tokenizers import AddedToken
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

Token = Union[Text, AddedToken]
TransformersTokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]


@TokenizerModule.register("code-tokenizer-fast", constructor="from_partial")
class CodeTokenizerModule(TransformerTokenizerModule):
    Params = TransformerTokenizerModule.Params

    def __init__(self, params: Params, **kwargs: Any) -> None:
        super().__init__(params, **kwargs)

        # add code special tokens
        for token in SpecialToken:
            self.special_tokens.append(token.value)

    def configure_tokenizer(
        self, tokenizer_path: Union[Text, Path], **kwargs: Any
    ) -> TransformersTokenizer:
        if isinstance(tokenizer_path, str):
            tokenizer_path = Path(tokenizer_path)
        vocab_file = str(tokenizer_path / "vocab.json")
        merges_file = str(tokenizer_path / "merges.txt")
        return CodeTokenizerFast(
            vocab_file=vocab_file, merges_file=merges_file, **kwargs
        )

    @classmethod
    def from_pretrained(cls, params: Params, **kwargs: Any) -> TransformersTokenizer:
        # prepare init arguments from params
        assert params.tokenizer_path is not None
        init_kwargs = cls.get_args(params)
        kwargs.update(init_kwargs)

        # get the pretrained tokenizer
        tokenizer = CodeTokenizerFast.from_pretrained(params.tokenizer_path, **kwargs)
        assert isinstance(tokenizer, PreTrainedTokenizerFast)

        return tokenizer

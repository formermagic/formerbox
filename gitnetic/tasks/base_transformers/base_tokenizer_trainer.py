from abc import abstractmethod
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, List, Optional, Text, Union

from tokenizers import AddedToken
from tokenizers.implementations import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from gitnetic.common.registrable import Registrable
from gitnetic.utils.utils import path_to_posix

from .base_tokenization import TransformerTokenizerFast

Token = Union[Text, AddedToken]
TransformersTokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]


def fix_tokenizer(tokenizer: TransformersTokenizer) -> None:
    init_kwargs = {}
    for key, value in tokenizer.init_kwargs.items():
        if isinstance(value, AddedToken):
            init_kwargs[key] = str(value)
        else:
            init_kwargs[key] = value

    tokenizer.init_kwargs = init_kwargs


class TokenizerTrainer(Registrable):
    @abstractmethod
    def train(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError()

    @abstractmethod
    def save_pretrained(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def add_argparse_args(parent_parser: ArgumentParser) -> ArgumentParser:
        raise NotImplementedError()


@TokenizerTrainer.register(
    name="transformer-tokenizer-trainer", constructor="from_args"
)
class TransformerTokenizerTrainer(TokenizerTrainer):
    def __init__(
        self,
        add_prefix_space: bool = False,
        lowercase: bool = False,
        dropout: Optional[float] = None,
        unicode_normalizer: Optional[Text] = None,
        continuing_subword_prefix: Optional[Text] = None,
        end_of_word_suffix: Optional[Text] = None,
        trim_offsets: bool = False,
    ) -> None:
        # pylint: disable=too-many-arguments
        self.add_prefix_space = add_prefix_space
        self.lowercase = lowercase
        self.dropout = dropout
        self.trim_offsets = trim_offsets
        self.special_tokens: List[Token] = [
            "<s>",
            "<pad>",
            "</s>",
            "<unk>",
            "<mask>",
        ]

        self.tokenizer = ByteLevelBPETokenizer(
            add_prefix_space=add_prefix_space,
            lowercase=lowercase,
            dropout=dropout,
            unicode_normalizer=unicode_normalizer,
            continuing_subword_prefix=continuing_subword_prefix,
            end_of_word_suffix=end_of_word_suffix,
            trim_offsets=trim_offsets,
        )

    def train(
        self,
        files: List[Text],
        vocab_size: int,
        min_frequency: int = 2,
        special_tokens: List[Token] = [],
        **extras: Any,
    ) -> None:
        # pylint: disable=dangerous-default-value, arguments-differ
        # pylint: disable=too-many-arguments
        del extras  # use only required params

        # prepare special tokens with an initial set
        special_tokens = self.special_tokens + special_tokens

        # train a tokenizer model
        self.tokenizer.train(
            files,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens,
        )

    def save_pretrained(
        self, tokenizer_output_path: Union[Text, Path], **extras: Any
    ) -> None:
        # pylint: disable=arguments-differ
        del extras  # use only required params
        # make sure the `tokenizer_output_path` is a pathlike object
        if isinstance(tokenizer_output_path, str):
            tokenizer_output_path = Path(tokenizer_output_path)

        # save the trained tokenizer to `tokenizer_output_path`
        save_dir = path_to_posix(tokenizer_output_path)
        self.tokenizer.save_model(save_dir)

        # prepare the pre-trained tokenizer
        fast_tokenizer = TransformerTokenizerFast(
            vocab_file=path_to_posix(tokenizer_output_path / "vocab.json"),
            merges_file=path_to_posix(tokenizer_output_path / "merges.txt"),
            add_prefix_space=self.add_prefix_space,
            trim_offsets=self.trim_offsets,
            lowercase=self.lowercase,
            dropout=self.dropout,
        )

        # save the pre-trained tokenizer
        fix_tokenizer(fast_tokenizer)
        fast_tokenizer.save_pretrained(save_dir)

    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # fmt: off
        parser.add_argument("--files", type=str, nargs="+", default=None, required=True,
                            help="")
        parser.add_argument("--vocab_size", type=int, default=None, required=True,
                            help="")
        parser.add_argument("--tokenizer_output_path", type=str, default=None, required=True,
                            help="")
        # fmt: on
        return parser

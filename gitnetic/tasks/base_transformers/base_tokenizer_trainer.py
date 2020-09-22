from argparse import ArgumentParser
from pathlib import Path
from typing import Any, List, Optional, Text, Union

from tokenizers import AddedToken
from tokenizers.implementations import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from typing_extensions import Literal

from gitnetic.utils.utils import path_to_posix

from .base_tokenization import TransformerTokenizerFast
from .tokenization_module import TokenizerModule

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


@TokenizerModule.register(name="transformer-tokenizer-fast", constructor="from_args")
class TransformerTokenizerModule(TokenizerModule):
    def __init__(
        self,
        add_prefix_space: bool = False,
        lowercase: bool = False,
        dropout: Optional[float] = None,
        unicode_normalizer: Optional[Text] = None,
        continuing_subword_prefix: Optional[Text] = None,
        end_of_word_suffix: Optional[Text] = None,
        trim_offsets: bool = False,
        **kwargs: Any,
    ) -> None:
        # pylint: disable=too-many-arguments
        super().__init__(**kwargs)

        self.special_tokens: List[Token] = [
            "<s>",
            "<pad>",
            "</s>",
            "<unk>",
            "<mask>",
        ]

        self.backend_tokenizer = ByteLevelBPETokenizer(
            add_prefix_space=add_prefix_space,
            lowercase=lowercase,
            dropout=dropout,
            unicode_normalizer=unicode_normalizer,
            continuing_subword_prefix=continuing_subword_prefix,
            end_of_word_suffix=end_of_word_suffix,
            trim_offsets=trim_offsets,
        )

    def configure_tokenizer(
        self, tokenizer_path: Union[Text, Path], **kwargs: Any
    ) -> PreTrainedTokenizerFast:
        # pylint: disable=arguments-differ
        if isinstance(tokenizer_path, str):
            tokenizer_path = Path(tokenizer_path)
        vocab_file = path_to_posix(tokenizer_path / "vocab.json")
        merges_file = path_to_posix(tokenizer_path / "merges.txt")
        return TransformerTokenizerFast(
            vocab_file=vocab_file, merges_file=merges_file, **kwargs
        )

    def train_tokenizer(
        self,
        files: List[Text],
        vocab_size: int,
        min_frequency: int = 2,
        special_tokens: Optional[List[Token]] = None,
        **extras: Any,
    ) -> None:
        # pylint: disable=arguments-differ
        del extras  # use designated args
        # prepare special tokens with an initial set
        special_tokens = self.special_tokens + (special_tokens or [])
        # train a tokenizer model
        self.backend_tokenizer.train(
            files,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens,
        )

    def save_pretrained(self, tokenizer_path: Union[Text, Path], **kwargs: Any) -> None:
        # pylint: disable=arguments-differ
        # make sure the `tokenizer_output_path` is a pathlike object
        if isinstance(tokenizer_path, str):
            tokenizer_path = Path(tokenizer_path)

        # save the trained tokenizer to `tokenizer_output_path`
        save_dir = path_to_posix(tokenizer_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        self.backend_tokenizer.save_model(save_dir)

        # prepare the pre-trained tokenizer
        tokenizer = self.configure_tokenizer(tokenizer_path=tokenizer_path, **kwargs)

        # save the pre-trained tokenizer
        fix_tokenizer(tokenizer)
        tokenizer.save_pretrained(save_dir)

    @staticmethod
    def from_pretrained(*args: Any, **kwargs: Any) -> PreTrainedTokenizerFast:
        tokenizer = TransformerTokenizerFast.from_pretrained(*args, **kwargs)
        assert isinstance(tokenizer, PreTrainedTokenizerFast)
        return tokenizer

    @staticmethod
    def add_argparse_args(
        parent_parser: ArgumentParser, stage: Literal["train", "tokenize"]
    ) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        if stage == "train":
            # fmt: off
            parser.add_argument("--files", type=str, nargs="+", default=None, required=True,
                                help="")
            parser.add_argument("--vocab_size", type=int, default=None, required=True,
                                help="")
            # fmt: on

        # fmt: off
        parser.add_argument("--tokenizer_path", type=str, default=None, required=True,
                            help="A path to pretrained tokenizer files or a save directory.")
        parser.add_argument("--add_prefix_space", type=bool, default=False, required=False,
                            help="Whether to add a leading space to the first word.")
        parser.add_argument("--trim_offsets", type=bool, default=True, required=False,
                            help="Whether the post processing step should trim " 
                                "offsets to avoid including whitespaces.")
        parser.add_argument("--lowercase", type=bool, default=False, required=False,
                            help="Whether to preprocess text as lowercase.")
        # fmt: on

        return parser

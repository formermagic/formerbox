import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Text, Union

from tokenizers import AddedToken
from tokenizers.implementations import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from gitnetic.common.dataclass_argparse import DataclassArgumentParser, DataclassBase
from gitnetic.utils.code_tokenizer import SpecialToken
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
    # pylint: disable=arguments-differ
    @dataclass
    class Params(DataclassBase):
        files: Optional[List[Text]] = field(
            default=None,
            metadata={"help": ""},
        )
        vocab_size: Optional[int] = field(
            default=None,
            metadata={"help": ""},
        )
        min_frequency: int = field(
            default=2,
            metadata={"help": ""},
        )
        tokenizer_path: Optional[Text] = field(
            default=None,
            metadata={
                "help": "A path to pretrained tokenizer files or a save directory."
            },
        )
        add_prefix_space: bool = field(
            default=False,
            metadata={"help": "Whether to add a leading space to the first word."},
        )
        trim_offsets: bool = field(
            default=True,
            metadata={
                "help": (
                    "Whether the post processing step should trim"
                    " offsets to avoid including whitespaces."
                )
            },
        )
        lowercase: bool = field(
            default=False,
            metadata={"help": "Whether to preprocess text as lowercase."},
        )
        dropout: Optional[float] = field(
            default=None,
            metadata={"help": ""},
        )
        unicode_normalizer: Optional[Text] = field(
            default=None,
            metadata={"help": ""},
        )
        continuing_subword_prefix: Optional[Text] = field(
            default=None,
            metadata={"help": ""},
        )
        end_of_word_suffix: Optional[Text] = field(
            default=None,
            metadata={"help": ""},
        )

    def __init__(self, params: Params, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.params = params
        self.special_tokens: List[Token] = [
            "<s>",
            "<pad>",
            "</s>",
            "<unk>",
            "<mask>",
        ]

        for token in SpecialToken:
            self.special_tokens.append(token.value)

        self.backend_tokenizer = ByteLevelBPETokenizer(
            add_prefix_space=params.add_prefix_space,
            lowercase=params.lowercase,
            dropout=params.dropout,
            unicode_normalizer=params.unicode_normalizer,
            continuing_subword_prefix=params.continuing_subword_prefix,
            end_of_word_suffix=params.end_of_word_suffix,
            trim_offsets=params.trim_offsets,
        )

    def configure_tokenizer(
        self, tokenizer_path: Union[Text, Path], **kwargs: Any
    ) -> PreTrainedTokenizerFast:
        if isinstance(tokenizer_path, str):
            tokenizer_path = Path(tokenizer_path)
        vocab_file = path_to_posix(tokenizer_path / "vocab.json")
        merges_file = path_to_posix(tokenizer_path / "merges.txt")
        return TransformerTokenizerFast(
            vocab_file=vocab_file, merges_file=merges_file, **kwargs
        )

    def train_tokenizer(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs  # use designated args
        # train a tokenizer model
        assert self.params.files is not None
        assert self.params.vocab_size is not None
        self.backend_tokenizer.train(
            files=self.params.files,
            vocab_size=self.params.vocab_size,
            min_frequency=self.params.min_frequency,
            special_tokens=self.special_tokens,
        )

    def save_pretrained(self, *args: Any, **kwargs: Any) -> None:
        del args  # use designated args
        # make sure the `tokenizer_output_path` is a pathlike object
        assert self.params.tokenizer_path is not None
        if isinstance(self.params.tokenizer_path, str):
            tokenizer_path = Path(self.params.tokenizer_path)
        else:
            tokenizer_path = self.params.tokenizer_path

        # save the trained tokenizer to `tokenizer_output_path`
        save_dir = path_to_posix(self.params.tokenizer_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        self.backend_tokenizer.save_model(save_dir)

        # prepare the pre-trained tokenizer
        tokenizer = self.configure_tokenizer(tokenizer_path=tokenizer_path, **kwargs)

        # save the pre-trained tokenizer
        fix_tokenizer(tokenizer)
        tokenizer.save_pretrained(save_dir)

    @staticmethod
    def from_pretrained(params: Params, **kwargs: Any) -> PreTrainedTokenizerFast:
        assert params.tokenizer_path is not None
        non_positional_params = vars(params).copy()
        non_positional_params.pop("tokenizer_path")
        kwargs.update(non_positional_params)

        tokenizer = TransformerTokenizerFast.from_pretrained(
            params.tokenizer_path, **kwargs
        )

        assert isinstance(tokenizer, PreTrainedTokenizerFast)

        return tokenizer

    @classmethod
    def add_argparse_args(cls, parser: DataclassArgumentParser) -> None:
        parser.add_arguments(cls.Params)

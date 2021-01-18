import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Text, Union

from formerbox.common import MISSING
from formerbox.data.tokenizers.tokenization_gpt2 import GPT2Tokenizer
from formerbox.data.tokenizers.tokenization_trainer import (
    TokenizerTrainerBase,
    TokenizerTrainerParams,
)
from formerbox.modules import TokenizerTrainer

from tokenizers.implementations import ByteLevelBPETokenizer

logger = logging.getLogger(__name__)


@TokenizerTrainer.register(name="gpt2", constructor="from_partial")
class GPT2TokenizerTrainer(TokenizerTrainerBase):
    # pylint: disable=arguments-differ
    @dataclass
    class Params(TokenizerTrainerParams):
        legacy_format: bool = field(
            default=False,
            metadata={
                "help": "Whether to save the tokenizer in legacy format,"
                " i.e. with tokenizer specific vocabulary and separate added_tokens files"
                " or in the unified JSON file format of the `tokenizers` library (default).",
            },
        )
        save_directory: Text = field(
            default=MISSING,
            metadata={
                "help": "A path for saving the pre-trained tokenizer.",
            },
        )
        add_prefix_space: bool = field(
            default=False,
            metadata={
                "help": "Whether to add a leading space to the first word.",
            },
        )
        trim_offsets: bool = field(
            default=True,
            metadata={
                "help": "Whether the post processing step should trim"
                " offsets to avoid including whitespaces.",
            },
        )
        lowercase: bool = field(
            default=False,
            metadata={
                "help": "Whether or not to preprocess text in lowercase.",
            },
        )
        dropout: Optional[float] = field(
            default=None,
            metadata={
                "help": "The likelihood of dropping a subword during calculating the frequency.",
            },
        )
        unicode_normalizer: Optional[Text] = field(
            default=None,
            metadata={
                "help": "The unicode text normalizer. Default is set to `None`.",
            },
        )
        continuing_subword_prefix: Optional[Text] = field(
            default=None,
            metadata={
                "help": "The subword prefix used for decoding the words. Default is set to `None`.",
            },
        )
        end_of_word_suffix: Optional[Text] = field(
            default=None,
            metadata={
                "help": "The suffix that comes after each word. Default is set to `None`.",
            },
        )

    params: Params
    params_type = Params

    def __init__(self, params: Params, **kwargs: Any) -> None:
        super().__init__(params, **kwargs)

    @classmethod
    def build_tokenizer(cls, params: Params) -> ByteLevelBPETokenizer:
        return ByteLevelBPETokenizer(
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
    ) -> GPT2Tokenizer:
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
        return GPT2Tokenizer(
            vocab_file=vocab_file,
            merges_file=merges_file,
            tokenizer_file=tokenizer_file,
            **kwargs,
        )

    def train_tokenizer(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs  # use designated args

        # train a tokenizer model
        assert self.params.files is not None
        assert self.params.vocab_size is not None
        assert isinstance(self.tokenizer, ByteLevelBPETokenizer)

        # add special tokens specific to the configured tokenizer
        self.tokenizer.add_special_tokens(self.special_tokens)
        # add additional tokens to include in the vocabulary
        self.tokenizer.add_tokens(self.additional_tokens)

        self.tokenizer.train(
            files=self.params.files,
            vocab_size=self.params.vocab_size,
            min_frequency=self.params.min_frequency,
            special_tokens=self.special_tokens,
        )

    def save_pretrained(
        self, save_directory: Optional[Text] = None, **kwargs: Any
    ) -> None:
        # take the directory from params if not specified
        if save_directory is None:
            assert self.params.save_directory is not None
            save_directory = self.params.save_directory

        # get the `legacy_format` argument value
        legacy_format = kwargs.pop("legacy_format", self.params.legacy_format)

        # save the pretrained tokenizer to `save_directory`
        super().save_pretrained(
            save_directory=save_directory,
            legacy_format=legacy_format,
            **kwargs,
        )

    @classmethod
    def get_tokenizer_args(cls, params: Params) -> Dict[Text, Any]:
        # prepare a copy of args for the pretrained tokenizer
        kwargs = vars(params).copy()

        # preserve pretrained tokenizer path
        kwargs.pop("tokenizer_path", None)

        # remove the legacy pretrained format flag
        kwargs.pop("legacy_format", None)

        # remove training-stage arguments
        kwargs.pop("files", None)
        kwargs.pop("vocab_size", None)
        kwargs.pop("min_frequency", None)
        kwargs.pop("dropout", None)
        kwargs.pop("unicode_normalizer", None)
        kwargs.pop("continuing_subword_prefix", None)
        kwargs.pop("end_of_word_suffix", None)
        kwargs.pop("save_directory", None)

        return kwargs

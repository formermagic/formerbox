import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Text, Union

from formerbox.common.dataclass_argparse import DataclassBase
from formerbox.modules.tokenizer_module import ParamsType, TokenizerModule
from formerbox.tasks.transformer_tokenization import ByteLevelBPETokenizerFast
from tokenizers import AddedToken
from tokenizers.implementations import BaseTokenizer, ByteLevelBPETokenizer
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

Token = Union[Text, AddedToken]
TransformersTokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

logger = logging.getLogger(__name__)


SPECIAL_TOKENS: List[Token] = [
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
]


class TransformerTokenizerModule(TokenizerModule[ParamsType]):
    special_tokens: List[Token]
    tokenizer: BaseTokenizer

    def __init__(self, params: ParamsType, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.params = params
        self.special_tokens: List[Token] = SPECIAL_TOKENS
        self.tokenizer = self.build_tokenizer(params)

    @classmethod
    def build_tokenizer(cls, params: ParamsType) -> BaseTokenizer:
        raise NotImplementedError()

    @classmethod
    def get_tokenizer_args(cls, params: ParamsType) -> Dict[Text, Any]:
        raise NotImplementedError()

    def configure_tokenizer(
        self, tokenizer_path: Union[Text, Path], **kwargs: Any
    ) -> TransformersTokenizer:
        raise NotImplementedError()

    def train_tokenizer(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError()

    def save_pretrained(
        self, save_directory: Text, legacy_format: bool, **kwargs: Any
    ) -> None:
        # make sure the `tokenizer_output_path` is a pathlike object
        tokenizer_path = Path(save_directory)

        # make the output dir if it doesn't exist
        tokenizer_path.mkdir(parents=True, exist_ok=True)

        # save the trained tokenizer to `tokenizer_output_path`
        self.tokenizer.save(str(tokenizer_path / "tokenizer.json"))

        # prepare the pre-trained tokenizer
        tokenizer = self.configure_tokenizer(tokenizer_path=tokenizer_path, **kwargs)

        # workaround for saving tokenizer bugs in the transformers backend
        self.__fix_tokenizer(tokenizer)

        # save the pre-trained tokenizer
        tokenizer.save_pretrained(
            save_directory=save_directory, legacy_format=legacy_format
        )

    @staticmethod
    def from_pretrained(
        tokenizer_path: Union[Text, Path], **kwargs: Any
    ) -> TransformersTokenizer:
        raise NotImplementedError()

    def __fix_tokenizer(self, tokenizer: TransformersTokenizer) -> None:
        init_kwargs = getattr(tokenizer, "init_kwargs", {})
        for key, value in init_kwargs.items():
            if isinstance(value, AddedToken):
                init_kwargs[key] = str(value)
            else:
                init_kwargs[key] = value


@TokenizerModule.register(name="byte-level-bpe-tokenizer", constructor="from_partial")
class ByteLevelBPETokenizerModule(TransformerTokenizerModule):
    # pylint: disable=arguments-differ
    @dataclass
    class Params(DataclassBase):
        files: Optional[List[Text]] = field(
            default=None,
            metadata={"help": "The input text files to train a tokenizer on."},
        )
        vocab_size: Optional[int] = field(
            default=None,
            metadata={"help": "The size of a trained tokenizer's vocabulary."},
        )
        min_frequency: int = field(
            default=2,
            metadata={"help": "The min frequency for calculating subwords merges."},
        )
        legacy_format: bool = field(
            default=True,
            metadata={
                "help": "Whether to save the tokenizer in legacy format (default),"
                " i.e. with tokenizer specific vocabulary and separate added_tokens files"
                " or in the unified JSON file format of the `tokenizers` library."
            },
        )
        save_directory: Optional[Text] = field(
            default=None,
            metadata={"help": "A path for saving the pre-trained tokenizer."},
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
            metadata={"help": "Whether or not to preprocess text in lowercase."},
        )
        dropout: Optional[float] = field(
            default=None,
            metadata={
                "help": "The likelihood of dropping a subword during calculating the frequency."
            },
        )
        unicode_normalizer: Optional[Text] = field(
            default=None,
            metadata={"help": "The unicode text normalizer. Default is set to `None`."},
        )
        continuing_subword_prefix: Optional[Text] = field(
            default=None,
            metadata={
                "help": "The subword prefix used for decoding the words. Default is set to `None`."
            },
        )
        end_of_word_suffix: Optional[Text] = field(
            default=None,
            metadata={
                "help": "The suffix that comes after each word. Default is set to `None`."
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
    ) -> ByteLevelBPETokenizerFast:
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
        return ByteLevelBPETokenizerFast(
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
    def from_pretrained(
        cls, tokenizer_path: Union[Text, Path], **kwargs: Any
    ) -> ByteLevelBPETokenizerFast:
        # convert `tokenizer_path` to string
        if isinstance(tokenizer_path, Path):
            tokenizer_path = str(tokenizer_path)

        # load the pretrained tokenizer
        tokenizer = ByteLevelBPETokenizerFast.from_pretrained(tokenizer_path, **kwargs)

        return tokenizer

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

        return kwargs

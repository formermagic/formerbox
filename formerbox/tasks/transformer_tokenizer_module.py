from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Text, Union

from formerbox.common.dataclass_argparse import DataclassBase
from formerbox.modules import TokenizerModule
from formerbox.tasks.transformer_tokenization import TransformerTokenizerFast
from formerbox.utils.utils import path_to_posix
from tokenizers import AddedToken
from tokenizers.implementations import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

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


@TokenizerModule.register(name="transformer-tokenizer-fast", constructor="from_partial")
class TransformerTokenizerModule(TokenizerModule):
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
        super().__init__(**kwargs)
        self.params = params
        self.special_tokens: List[Token] = [
            "<s>",
            "<pad>",
            "</s>",
            "<unk>",
            "<mask>",
        ]

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
    ) -> TransformersTokenizer:
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

    def save_pretrained(
        self, save_directory: Optional[Text] = None, **kwargs: Any
    ) -> None:
        # take the directory from params if not specified
        if save_directory is None:
            assert self.params.tokenizer_path is not None
            save_directory = self.params.tokenizer_path

        # make sure the `tokenizer_output_path` is a pathlike object
        if isinstance(save_directory, str):
            tokenizer_path = Path(save_directory)
        else:
            tokenizer_path = save_directory

        # make the output dir if it doesn't exist
        tokenizer_path.mkdir(exist_ok=True)

        # save the trained tokenizer to `tokenizer_output_path`
        self.backend_tokenizer.save_model(save_directory)

        # prepare the pre-trained tokenizer
        tokenizer = self.configure_tokenizer(tokenizer_path=tokenizer_path, **kwargs)

        # workaround for saving tokenizer bugs in the transformers backend
        fix_tokenizer(tokenizer)
        # save the pre-trained tokenizer
        tokenizer.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, params: Params, **kwargs: Any) -> TransformersTokenizer:
        # prepare init arguments from params
        assert params.tokenizer_path is not None
        init_kwargs = cls.get_args(params)
        kwargs.update(init_kwargs)

        # get the pretrained tokenizer
        tokenizer = TransformerTokenizerFast.from_pretrained(
            params.tokenizer_path, **kwargs
        )
        assert isinstance(tokenizer, PreTrainedTokenizerFast)

        return tokenizer

    @staticmethod
    def get_args(params: Params) -> Dict[Text, Any]:
        # prepare a copy of args for the pretrained tokenizer
        kwargs = vars(params).copy()

        # preserve pretrained tokenizer path
        kwargs.pop("tokenizer_path", None)

        # remove training-stage arguments
        kwargs.pop("files", None)
        kwargs.pop("vocab_size", None)
        kwargs.pop("min_frequency", None)

        return kwargs

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Text, Type, Union

from formerbox.common.dataclass_argparse import MISSING, DataclassBase
from formerbox.modules.tokenizer_trainer import ParamsType, TokenizerTrainer
from transformers import PreTrainedTokenizerFast as Tokenizer

from tokenizers import AddedToken
from tokenizers.implementations import BaseTokenizer

Token = Union[Text, AddedToken]

logger = logging.getLogger(__name__)


SPECIAL_TOKENS: List[Token] = [
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
]

VERY_LARGE_INTEGER = int(1e30)


@dataclass
class TokenizerTrainerParams(DataclassBase):
    files: List[Text] = field(
        default_factory=MISSING,
        metadata={"help": "The input text files to train a tokenizer on."},
    )
    vocab_size: int = field(
        default=MISSING,
        metadata={"help": "The size of a trained tokenizer's vocabulary."},
    )
    min_frequency: int = field(
        default=2,
        metadata={"help": "The min frequency for calculating subwords merges."},
    )
    model_max_length: int = field(
        default=VERY_LARGE_INTEGER,
        metadata={"help": "The maximum input length for the associated model."},
    )


class TokenizerTrainerBase(TokenizerTrainer[ParamsType]):
    params: ParamsType
    params_type: Type[ParamsType]
    additional_tokens: List[Token]
    special_tokens: List[Token]
    tokenizer: BaseTokenizer

    def __init__(self, params: ParamsType, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.params = params
        self.additional_tokens = []
        self.special_tokens = SPECIAL_TOKENS
        self.tokenizer = self.build_tokenizer(params)

    @classmethod
    def build_tokenizer(cls, params: ParamsType) -> BaseTokenizer:
        raise NotImplementedError()

    @classmethod
    def get_tokenizer_args(cls, params: ParamsType) -> Dict[Text, Any]:
        raise NotImplementedError()

    def configure_tokenizer(
        self, tokenizer_path: Union[Text, Path], **kwargs: Any
    ) -> Tokenizer:
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

        # save the legacy tokenizer to `tokenizer_path`
        if legacy_format:
            self.tokenizer.save_model(str(tokenizer_path))
        # save the unified tokenizer to `tokenizer_path`
        else:
            self.tokenizer.save(str(tokenizer_path / "tokenizer.json"))

        # prepare the pre-trained tokenizer
        tokenizer = self.configure_tokenizer(tokenizer_path=tokenizer_path, **kwargs)

        # save the pre-trained tokenizer
        tokenizer.save_pretrained(
            save_directory=save_directory, legacy_format=legacy_format
        )

import logging
from pathlib import Path
from typing import Any, Dict, List, Text, Union

from formerbox.modules.tokenizer_trainer import ParamsType, TokenizerTrainer
from tokenizers import AddedToken
from tokenizers.implementations import BaseTokenizer
from transformers import PreTrainedTokenizerFast as Tokenizer

Token = Union[Text, AddedToken]

logger = logging.getLogger(__name__)


SPECIAL_TOKENS: List[Token] = [
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
]


class TokenizerTrainerBase(TokenizerTrainer[ParamsType]):
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

        # workaround for saving tokenizer bugs in the transformers backend
        self.__fix_tokenizer(tokenizer)

        # save the pre-trained tokenizer
        tokenizer.save_pretrained(
            save_directory=save_directory, legacy_format=legacy_format
        )

    def __fix_tokenizer(self, tokenizer: Tokenizer) -> None:
        init_kwargs = getattr(tokenizer, "init_kwargs", {})
        for key, value in init_kwargs.items():
            if isinstance(value, AddedToken):
                init_kwargs[key] = str(value)
            else:
                init_kwargs[key] = value

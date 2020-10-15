from abc import ABCMeta, abstractmethod
from typing import Any, List, Optional, Text, Type, Union

from formerbox.common.has_params import HasParsableParams, ParamsType
from formerbox.common.registrable import Registrable
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers.tokenization_utils_base import (
    BatchEncoding,
    PaddingStrategy,
    TruncationStrategy,
)

Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]


class TokenizerModule(Registrable, HasParsableParams[ParamsType], metaclass=ABCMeta):
    params: ParamsType
    params_type: Type[ParamsType]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.tokenizer: Optional[Tokenizer] = None
        self.args = args
        self.kwargs = kwargs

    @abstractmethod
    def configure_tokenizer(self, *args: Any, **kwargs: Any) -> Tokenizer:
        raise NotImplementedError()

    @abstractmethod
    def train_tokenizer(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError()

    @abstractmethod
    def save_pretrained(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def from_pretrained(*args: Any, **kwargs: Any) -> Tokenizer:
        raise NotImplementedError()

    def __call__(self, args: Any, **kwargs: Any) -> BatchEncoding:
        return self.tokenizer.__call__(*args, **kwargs)

    def tokenize(
        self, text: str, pair: Optional[str] = None, add_special_tokens: bool = False
    ) -> List[Text]:
        return self.tokenizer.tokenize(
            text=text, pair=pair, add_special_tokens=add_special_tokens
        )

    def convert_ids_to_tokens(
        self, ids: Union[int, List[int]], skip_special_tokens: bool = False
    ) -> Union[Text, List[Text]]:
        return self.tokenizer.convert_ids_to_tokens(
            ids=ids, skip_special_tokens=skip_special_tokens
        )  # type: ignore

    def convert_tokens_to_ids(
        self, tokens: Union[Text, List[Text]]
    ) -> Union[int, List[int]]:
        return self.tokenizer.convert_tokens_to_ids(tokens=tokens)

    def set_truncation_and_padding(
        self,
        padding_strategy: PaddingStrategy,
        truncation_strategy: TruncationStrategy,
        max_length: int,
        stride: int,
        pad_to_multiple_of: Optional[int],
    ) -> None:
        if isinstance(self.tokenizer, PreTrainedTokenizerFast):
            self.tokenizer.set_truncation_and_padding(
                padding_strategy=padding_strategy,
                truncation_strategy=truncation_strategy,
                max_length=max_length,
                stride=stride,
                pad_to_multiple_of=pad_to_multiple_of,
            )

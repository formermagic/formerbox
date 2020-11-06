from abc import ABCMeta, abstractmethod
from typing import Any, Optional, Type

from formerbox.common.has_params import HasParsableParams, ParamsType
from formerbox.common.registrable import Registrable
from transformers import PreTrainedTokenizerFast as Tokenizer


class TokenizerTrainer(Registrable, HasParsableParams[ParamsType], metaclass=ABCMeta):
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

from abc import abstractmethod
from argparse import ArgumentParser
from typing import Any, Type

from transformers import PreTrainedTokenizerFast

from gitnetic.common.registrable import Registrable


class TokenizerTrainer(Registrable):
    def __init__(
        self, tokenizer_module_cls: Type["TokenizerFastModule"], **kwargs: Any
    ) -> None:
        self.tokenizer_module_cls = tokenizer_module_cls
        self.kwargs = kwargs

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


class TokenizerFastModule(Registrable, PreTrainedTokenizerFast):
    @property
    @abstractmethod
    def trainer_cls(self) -> Type[TokenizerTrainer]:
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def add_argparse_args(parent_parser: ArgumentParser) -> ArgumentParser:
        raise NotImplementedError()

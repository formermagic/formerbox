from abc import abstractmethod
from argparse import ArgumentParser
from typing import Any, Text, Type, TypeVar, Union

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from gitnetic.common.registrable import Registrable

from .base import TrainingParams
from .base_config import model_from_config, tokenizer_from_config
from .base_modules import TransformerDataModule, TransformerModule

T = TypeVar("T", bound="TaskModule")
Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]


class TaskModule(Registrable):
    def __init__(
        self,
        tokenizer: Tokenizer,
        module: TransformerModule,
        datamodule: TransformerDataModule,
    ) -> None:
        self.tokenizer = tokenizer
        self.module = module
        self.datamodule = datamodule

    @classmethod
    @abstractmethod
    def setup(cls: Type[T], *args: Any, **kwargs: Any) -> T:
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def add_argparse_args(parent_parser: ArgumentParser) -> ArgumentParser:
        raise NotImplementedError()


@TaskModule.register("transformer-task")
class TransformerTask(TaskModule):
    @classmethod
    def setup(cls, *args: Any, **kwargs: Any) -> "TransformerTask":
        del args  # use designated args

        # prepare the pretrained tokenizer
        config_path: Text = kwargs["config_path"]
        tokenizer_path: Text = kwargs["tokenizer_path"]
        tokenizer = tokenizer_from_config(config_path, tokenizer_path)
        assert isinstance(tokenizer, Tokenizer.__args__)  # type: ignore

        # prepare a model to train
        model = model_from_config(
            config_path,
            vocab_size=tokenizer.vocab_size,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        # prepare a transformer module
        training_params = TrainingParams.from_args(**kwargs)
        module = TransformerModule(model, tokenizer, training_params)

        # prepare a transformer datamodule
        datamodule = TransformerDataModule.from_args(tokenizer=tokenizer, **kwargs)

        return cls(tokenizer, module, datamodule)

    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # fmt: off
        parser.add_argument("--config_path", type=str, default=None, required=True,
                            help="A path to the file with model and tokenizer configs.")
        parser.add_argument("--tokenizer_path", type=str, default=None, required=True,
                            help="A path to the dir with saved pretrained tokenizer.")
        # fmt: on

        # parser = TransformerTrainer.add_argparse_args(parser)
        parser = TransformerDataModule.add_argparse_args(parser)
        parser = TransformerModule.add_argparse_args(parser)

        return parser

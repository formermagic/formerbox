from abc import ABCMeta, abstractmethod
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Optional, Text, Tuple, Type, TypeVar, Union

from gitnetic.common.dataclass_argparse import DataclassArgumentParser, DataclassBase
from gitnetic.common.registrable import ArgumentRegistrable
from transformers import PreTrainedTokenizerBase
from typing_extensions import Protocol

from .base_config import model_from_config, tokenizer_from_config
from .base_modules import TransformerDataModule, TransformerModule

T = TypeVar("T", bound="TaskModule")  # pylint: disable=invalid-name
Tokenizer = PreTrainedTokenizerBase
ModuleType = TypeVar("ModuleType", bound="TransformerModule")
DataModuleType = TypeVar("DataModuleType", bound="TransformerDataModule")


class TaskModuleBase(Protocol[ModuleType, DataModuleType]):
    ...


class TaskModule(
    TaskModuleBase[ModuleType, DataModuleType],
    ArgumentRegistrable,
    metaclass=ABCMeta,
):
    def __init__(
        self,
        tokenizer: Tokenizer,
        module: ModuleType,
        datamodule: DataModuleType,
    ) -> None:
        self.tokenizer = tokenizer
        self.module = module
        self.datamodule = datamodule

    @classmethod
    @abstractmethod
    def setup(
        cls: Type[T],
        params: Tuple[Union[DataclassBase, Namespace], ...],
    ) -> T:
        raise NotImplementedError()


@TaskModule.register("transformer-task")
class TransformerTask(TaskModule[TransformerModule, TransformerDataModule]):
    @dataclass
    class Params(DataclassBase):
        config_path: Text = field(
            metadata={"help": "A path to the file with model and tokenizer configs."},
        )
        tokenizer_path: Text = field(
            metadata={"help": "A path to the dir with saved pretrained tokenizer."},
        )

    @classmethod
    def setup(
        cls: Type["TransformerTask"],
        params: Tuple[Union[DataclassBase, Namespace], ...],
    ) -> "TransformerTask":
        task_params: Optional[cls.Params] = None
        module_params: Optional[TransformerModule.Params] = None
        datamodule_params: Optional[TransformerDataModule.Params] = None

        for param in params:
            if isinstance(param, cls.Params):
                task_params = param
            elif isinstance(param, TransformerModule.Params):
                module_params = param
            elif isinstance(param, TransformerDataModule.Params):
                datamodule_params = param

        assert task_params is not None
        tokenizer = tokenizer_from_config(
            task_params.config_path, task_params.tokenizer_path
        )

        # prepare a model to train
        model = model_from_config(
            task_params.config_path,
            vocab_size=tokenizer.vocab_size,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        # prepare a transformer module
        assert module_params is not None
        module = TransformerModule(
            model=model, tokenizer=tokenizer, params=module_params
        )

        # prepare a transformer datamodule
        assert datamodule_params is not None
        datamodule = TransformerDataModule(
            tokenizer=tokenizer, params=datamodule_params
        )

        return cls(tokenizer, module, datamodule)

    @classmethod
    def add_argparse_args(
        cls: Type["TransformerTask"], parser: DataclassArgumentParser
    ) -> None:
        parser.add_arguments(cls.Params)
        TransformerModule.add_argparse_args(parser)
        TransformerDataModule.add_argparse_args(parser)

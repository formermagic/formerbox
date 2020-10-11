from abc import ABCMeta, abstractmethod
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Text, Tuple, Type, TypeVar, Union

from gitnetic.common.dataclass_argparse import (
    DataclassArgumentParser,
    DataclassBase,
    get_params_item,
)
from gitnetic.common.has_params import HasParsableParams
from gitnetic.common.registrable import Registrable
from transformers import PreTrainedTokenizerBase
from typing_extensions import Protocol

from .base_config import model_from_config, tokenizer_from_config
from .base_modules import TransformerDataModule, TransformerModule

T = TypeVar("T", bound="TaskModule")  # pylint: disable=invalid-name
Tokenizer = PreTrainedTokenizerBase
ModuleType = TypeVar("ModuleType", bound="TransformerModule")
DataModuleType = TypeVar("DataModuleType", bound="TransformerDataModule")
ParamType = Union[DataclassBase, Namespace]


class TaskModuleBase(Protocol[ModuleType, DataModuleType]):
    ...


class TaskModule(
    TaskModuleBase[ModuleType, DataModuleType],
    Registrable,
    HasParsableParams,
    metaclass=ABCMeta,
):
    def __init__(
        self,
        tokenizer: Tokenizer,
        module: ModuleType,
        datamodule: DataModuleType,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.module = module
        self.datamodule = datamodule

    @classmethod
    @abstractmethod
    def setup(cls: Type[T], params: Tuple[ParamType, ...]) -> T:
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

    params: Params
    params_type: Type[Params] = Params

    ComponentParams = Tuple[
        params_type,
        TransformerModule.params_type,
        TransformerDataModule.params_type,
    ]

    @classmethod
    def get_params(cls, params: Tuple[ParamType, ...]) -> "ComponentParams":
        # get the params for task components
        task_params = get_params_item(
            params=params,
            params_type=cls.params_type,
        )
        module_params = get_params_item(
            params=params,
            params_type=TransformerModule.params_type,
        )
        datamodule_params = get_params_item(
            params=params,
            params_type=TransformerDataModule.params_type,
        )

        # make sure the params exist
        assert task_params is not None
        assert module_params is not None
        assert datamodule_params is not None

        return task_params, module_params, datamodule_params

    @classmethod
    def setup(
        cls: Type["TransformerTask"], params: Tuple[ParamType, ...]
    ) -> "TransformerTask":
        # get the params for task components
        task_params, module_params, datamodule_params = cls.get_params(params)

        # prepare the tokenizer from config
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
        module = TransformerModule(
            model=model, tokenizer=tokenizer, params=module_params
        )

        # prepare a transformer datamodule
        datamodule = TransformerDataModule(
            tokenizer=tokenizer, params=datamodule_params
        )

        return cls(tokenizer, module, datamodule)

    @classmethod
    def add_argparse_params(
        cls: Type["TransformerTask"], parser: DataclassArgumentParser
    ) -> None:
        parser.add_arguments(cls.Params)
        TransformerModule.add_argparse_params(parser)
        TransformerDataModule.add_argparse_params(parser)

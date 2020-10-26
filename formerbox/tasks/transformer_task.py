import logging
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Text, Tuple, Type, TypeVar, Union

from formerbox.common.dataclass_argparse import (
    DataclassArgumentParser,
    DataclassBase,
    get_params_item,
)
from formerbox.modules import TransformerDataModule as DataModule
from formerbox.modules import TransformerModule as Module
from formerbox.tasks.task_module import TaskModule
from formerbox.training.load_from_config import model_from_config, tokenizer_from_config

T = TypeVar("T", bound="TaskModule")  # pylint: disable=invalid-name
ParamType = Union[DataclassBase, Namespace]

logger = logging.getLogger(__name__)


@TaskModule.register("transformer-task")
class TransformerTask(TaskModule[Module, DataModule]):
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
        Module.params_type,
        DataModule.params_type,
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
            params_type=Module.params_type,
        )
        datamodule_params = get_params_item(
            params=params,
            params_type=DataModule.params_type,
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
        module = Module(model=model, tokenizer=tokenizer, params=module_params)

        # prepare a transformer datamodule
        datamodule = DataModule(tokenizer=tokenizer, params=datamodule_params)

        return cls(tokenizer, module, datamodule)

    @classmethod
    def add_argparse_params(
        cls: Type["TransformerTask"], parser: DataclassArgumentParser
    ) -> None:
        parser.add_arguments(cls.Params)
        Module.add_argparse_params(parser)
        DataModule.add_argparse_params(parser)

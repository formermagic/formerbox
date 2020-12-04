import logging
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Text, Tuple, Type, Union

from formerbox.common.dataclass_argparse import (
    MISSING,
    DataclassArgumentParser,
    DataclassBase,
)
from formerbox.common.has_params import ParamsType
from formerbox.modules import TransformerDataModule as DataModule
from formerbox.modules import TransformerModule as Module
from formerbox.tasks.task_module import TaskModule
from formerbox.training.load_from_config import model_from_config, tokenizer_from_config
from transformers import PreTrainedTokenizerFast as Tokenizer

logger = logging.getLogger(__name__)


@TaskModule.register("transformer-task")
class TransformerTask(TaskModule[ParamsType]):
    @dataclass
    class Params(DataclassBase):
        config_path: Text = field(
            default=MISSING,
            metadata={"help": "A path to the file with model and tokenizer configs."},
        )
        tokenizer_path: Text = field(
            default=MISSING,
            metadata={"help": "A path to the dir with saved pretrained tokenizer."},
        )

    params: Params
    params_type: Type[Params] = Params

    def __init__(
        self,
        tokenizer: Tokenizer,
        module: Module,
        datamodule: DataModule,
    ) -> None:
        super().__init__(tokenizer, module, datamodule)
        self.tokenizer = tokenizer
        self.module = module
        self.datamodule = datamodule

    @classmethod
    def setup(
        cls: Type["TransformerTask"],
        params: Tuple[Union[DataclassBase, Namespace], ...],
    ) -> "TransformerTask":
        # prepare the tokenizer from config
        task_params = cls.get_params(params, cls.params_type)
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
        module_params = cls.get_params(params, Module.params_type)
        module = Module(model=model, tokenizer=tokenizer, params=module_params)

        # prepare a transformer datamodule
        datamodule_params = cls.get_params(params, DataModule.params_type)
        datamodule = DataModule(tokenizer=tokenizer, params=datamodule_params)

        return cls(tokenizer, module, datamodule)

    @classmethod
    def add_argparse_params(
        cls: Type["TransformerTask"], parser: DataclassArgumentParser
    ) -> None:
        parser.add_arguments(cls.params_type)
        Module.add_argparse_params(parser)
        DataModule.add_argparse_params(parser)

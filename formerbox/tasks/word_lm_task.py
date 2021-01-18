import logging
from argparse import Namespace
from dataclasses import dataclass
from typing import Tuple, Type, Union

from formerbox.common.dataclass_argparse import DataclassArgumentParser, DataclassBase
from formerbox.common.has_params import ParamsType
from formerbox.modules import WordLMDataModule as DataModule
from formerbox.modules import WordLMModule as Module
from formerbox.tasks.task_module import TaskModule, TaskParams
from formerbox.training.load_from_config import model_from_config, tokenizer_from_config
from transformers import PreTrainedTokenizerFast as Tokenizer

logger = logging.getLogger(__name__)


@TaskModule.register("word_lm")
class WordLMTask(TaskModule[ParamsType]):
    @dataclass
    class Params(TaskParams):
        pass

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
        cls: Type["WordLMTask"],
        params: Tuple[Union[DataclassBase, Namespace], ...],
    ) -> "WordLMTask":
        # prepare the tokenizer from config
        task_params = cls.get_params(params, cls.params_type)
        tokenizer = tokenizer_from_config(
            task_params.config_path, task_params.tokenizer_path
        )

        # prepare vocab size with or without added tokens
        vocab_size = tokenizer.backend_tokenizer.get_vocab_size(
            task_params.include_added_tokens
        )

        # prepare model max length for positional embeddings
        assert hasattr(tokenizer, "model_max_length")
        model_max_length = getattr(tokenizer, "model_max_length")

        # some models also add extra positions to positional embeddings
        assert tokenizer.pad_token_id is not None
        extra_pos_embeddings = tokenizer.pad_token_id + 1

        # prepare a model to train
        model = model_from_config(
            task_params.config_path,
            vocab_size=vocab_size,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_position_embeddings=model_max_length,
            extra_pos_embeddings=extra_pos_embeddings,
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
        cls: Type["WordLMTask"], parser: DataclassArgumentParser
    ) -> None:
        parser.add_arguments(cls.params_type)
        Module.add_argparse_params(parser)
        DataModule.add_argparse_params(parser)

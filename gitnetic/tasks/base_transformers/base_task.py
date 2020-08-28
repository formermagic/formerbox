from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Any, Dict, Text, Union

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from .base import TrainingParams
from .base_config import model_from_config, tokenizer_from_config
from .base_modules import TransformerDataModule, TransformerModule

Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]


@dataclass
class TransformerTask:
    tokenizer: Tokenizer
    module: TransformerModule
    datamodule: TransformerDataModule

    @classmethod
    def setup(cls, args: Dict[Text, Any]) -> "TransformerTask":
        # prepare the pretrained tokenizer
        config_path: Text = args["config_path"]
        tokenizer_path: Text = args["tokenizer_path"]
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
        training_params = TrainingParams.from_args(args)

        module = TransformerModule(model, tokenizer, training_params)

        # prepare a transformer datamodule
        datamodule = TransformerDataModule.from_args(args, tokenizer=tokenizer)

        return cls(tokenizer, module, datamodule)

    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # fmt: off
        parser.add_argument("--config_path", type=str, default=None, required=True,
                            help="")
        parser.add_argument("--tokenizer_path", type=str, default=None, required=True,
                            help="")

        # fmt: on

        # parser = TransformerTrainer.add_argparse_args(parser)
        parser = TransformerDataModule.add_argparse_args(parser)
        parser = TransformerModule.add_argparse_args(parser)

        return parser

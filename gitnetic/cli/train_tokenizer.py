import logging
import os
from argparse import Namespace, _SubParsersAction
from dataclasses import dataclass, field
from typing import Any, Dict, Text, Tuple, Union

from typeguard import typechecked

from gitnetic.cli.subcommand import Subcommand
from gitnetic.common.dataclass_argparse import (
    DataclassArgumentParser,
    DataclassBase,
    get_params_item,
)
from gitnetic.tasks.base_transformers import TokenizerModule

logger = logging.getLogger(__name__)


@Subcommand.register("train_tokenizer")
class TrainTokenizer(Subcommand):
    @dataclass
    class Params(DataclassBase):
        tokenizer: Text = field(
            metadata={
                "choices": sorted(TokenizerModule.list_available()),
                "help": "",
            },
        )

    def add_subparser(
        self, parser: _SubParsersAction
    ) -> Tuple[DataclassArgumentParser, Dict[Text, Any]]:
        description = """Train a tokenizer on text data files."""
        subparser = self._add_parser(
            parser,
            name=self.name,
            description=description,
            help=description,
        )

        # add command arguments
        subparser.add_arguments(self.Params)

        def add_dynamic_args(parser: DataclassArgumentParser) -> None:
            # get the parsed command arguments
            parsed_params = subparser.parse_args_into_dataclasses(
                return_remaining_strings=True
            )
            params = parsed_params[0]
            assert isinstance(params, self.Params)

            # add dybamic args to the subparser
            tokenizer_cls, _ = TokenizerModule.from_registry(params.tokenizer)
            tokenizer_cls.add_argparse_args(subparser)

            # inject dataclass_types to the parent parser
            parser.dataclass_types = subparser.dataclass_types

        defaults = dict(
            func=train_tokenizer,
            add_dynamic_args=add_dynamic_args,
        )

        return subparser, defaults


@typechecked
def train_tokenizer(params: Tuple[Union[DataclassBase, Namespace], ...]) -> None:
    # make sure tokenizer parallelizm is disabled
    # since it might cause deadlocks while preprocessing
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    cmd_params = get_params_item(params, params_type=TrainTokenizer.Params)

    tokenizer_cls, tokenizer_init = TokenizerModule.from_registry(cmd_params.tokenizer)
    tokenizer_params = get_params_item(params, params_type=tokenizer_cls.Params)
    tokenizer = tokenizer_init(params=tokenizer_params)

    tokenizer.train_tokenizer()

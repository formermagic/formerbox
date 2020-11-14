import logging
from argparse import Namespace
from dataclasses import dataclass, field
from time import time
from typing import Any, Dict, Text, Tuple, Union

from formerbox.cli.subcommand import Subcommand, _SubParsersAction
from formerbox.common.dataclass_argparse import (
    MISSING,
    DataclassArgumentParser,
    DataclassBase,
    get_params_item,
)
from formerbox.data.dataset_converter import DatasetConverter

logger = logging.getLogger(__name__)


@Subcommand.register("convert_dataset")
class ConvertDataset(Subcommand):
    @dataclass
    class Params(DataclassBase):
        converter: Text = field(
            default=MISSING,
            metadata={
                "choices": DatasetConverter,
                "help": "The name of a registered dataset converter to use.",
            },
        )

    def add_subparser(
        self, parser: _SubParsersAction
    ) -> Tuple[DataclassArgumentParser, Dict[Text, Any]]:
        description = (
            """Convert a raw dataset into a prepared for processing text dataset."""
        )
        subparser = self._add_parser(
            parser,
            name=self.name,
            description=description,
            help=description,
        )

        # get the type of a converter to use
        subparser.add_arguments(self.Params)

        def add_dynamic_args(parser: DataclassArgumentParser) -> None:
            # get the parsed command arguments
            parsed_params = subparser.parse_args_into_dataclasses(
                return_remaining_strings=True
            )
            params = parsed_params[0]
            assert isinstance(params, self.Params)

            # add dybamic args to the subparser
            converter_cls, _ = DatasetConverter.from_registry(params.converter)
            converter_cls.add_argparse_params(subparser)

            # inject dataclass_types to the parent parser
            parser.dataclass_types = subparser.dataclass_types

        defaults = dict(
            func=convert_dataset,
            add_dynamic_args=add_dynamic_args,
        )

        return subparser, defaults


def convert_dataset(params: Tuple[Union[DataclassBase, Namespace], ...]) -> None:
    cmd_params = get_params_item(params, params_type=ConvertDataset.Params)

    converter_cls, converter_init = DatasetConverter.from_registry(cmd_params.converter)
    converter_params = get_params_item(params, params_type=converter_cls.params_type)
    converter = converter_init(params=converter_params)

    start_time = time()
    converter.convert()
    time_delta = time() - start_time
    logger.info("Wall time: %.3fs", time_delta)

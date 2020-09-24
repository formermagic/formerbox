from argparse import Namespace
from time import time
from typing import Any, Dict, Text, Tuple, Union

from typeguard import typechecked

from gitnetic.cli.subcommand import Subcommand, _SubParsersAction
from gitnetic.common.dataclass_argparse import (
    DataclassArgumentParser,
    DataclassBase,
    get_parsed_attr,
)
from gitnetic.data.dataset_converter import DatasetConverter


@Subcommand.register("convert_dataset")
class ConvertDataset(Subcommand):
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
        choices = sorted(DatasetConverter.list_available())
        subparser.add_argument(
            "--converter_type",
            type=str,
            choices=choices,
            required=True,
            help="",
        )

        def add_dynamic_args(parser: DataclassArgumentParser) -> None:
            # add dybamic args to the subparser
            args = subparser.parse_known_args()[0]
            converter_cls, _ = DatasetConverter.from_registry(args.converter_type)
            converter_cls.add_argparse_args(subparser)
            # inject dataclass_types to the parent parser
            parser.dataclass_types = subparser.dataclass_types

        defaults = dict(
            func=convert_dataset,
            add_dynamic_args=add_dynamic_args,
        )

        return subparser, defaults


@typechecked
def convert_dataset(params: Tuple[Union[DataclassBase, Namespace], ...]) -> None:
    parser = converter_cls.add_argparse_args(parser)
    params = parser.parse_args_into_dataclasses()
    converter_type = get_parsed_attr(params, "converter_type")
    assert isinstance(converter_type, str)
    converter_cls = DatasetConverter.from_name(converter_type)
    converter = converter_cls(params=params[0])

    start = time()
    converter.convert()
    print(f"Time elapsed: {time() - start}")

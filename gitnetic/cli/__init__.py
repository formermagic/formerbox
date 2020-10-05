from argparse import Namespace
from typing import Optional, Text

from gitnetic.cli.convert_dataset import ConvertDataset
from gitnetic.cli.preprocess import Preprocess
from gitnetic.cli.subcommand import Subcommand
from gitnetic.cli.train import Train
from gitnetic.cli.train_tokenizer import TrainTokenizer
from gitnetic.common.dataclass_argparse import DataclassArgumentParser
from gitnetic.common.utils import import_module_and_submodules, import_user_module


def register_additional_classes(args: Namespace) -> None:
    for package_name in args.include_package:
        import_module_and_submodules(package_name)

    if args.user_dir is not None:
        module_name = import_user_module(args.user_dir)
        import_module_and_submodules(module_name)


def make_parser(prog: Optional[Text] = None) -> DataclassArgumentParser:
    # crete an argument parser with plugin-resolving args
    parser = DataclassArgumentParser(prog=prog)
    parser.add_argument(
        "--include-package",
        type=str,
        action="append",
        default=[],
        help="Additional packages to include.",
    )
    parser.add_argument(
        "--user-dir",
        type=str,
        default=None,
        help="A directory with user custom plugins to include.",
    )

    # register additional classes with specified packages or user-dir
    args = parser.parse_known_args()[0]
    register_additional_classes(args)

    # create subparsers for working with subcommands
    subparsers = parser.add_subparsers(
        title="Commands", parser_class=DataclassArgumentParser
    )

    # add subcommands and register their args
    for subcommand_name in sorted(Subcommand.list_available()):
        subcommand_cls = Subcommand.from_name(subcommand_name)
        subcommand = subcommand_cls()

        subparser, defaults = subcommand.add_subparser(subparsers)
        subparser.set_defaults(**defaults)

    return parser


def main(prog: Optional[Text] = None) -> None:
    parser = make_parser(prog)
    args = parser.parse_known_args()[0]

    # If a subparser is triggered, it adds its work as `args.func`.
    # So if no such attribute has been added, no subparser was triggered,
    # so give the user some help.
    if "func" in dir(args):
        if hasattr(args, "add_dynamic_args"):
            args.add_dynamic_args(parser)
        params = parser.parse_args_into_dataclasses()
        args.func(params)
    else:
        parser.print_help()

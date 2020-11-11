from argparse import Namespace, _SubParsersAction
from typing import Iterable, Optional, Text

from formerbox.cli.convert_dataset import ConvertDataset
from formerbox.cli.preprocess import Preprocess
from formerbox.cli.subcommand import Subcommand
from formerbox.cli.train import Train
from formerbox.cli.train_tokenizer import TrainTokenizer
from formerbox.common.dataclass_argparse import DataclassArgumentParser
from formerbox.common.utils import import_module_and_submodules, import_user_module


def register_additional_classes(args: Namespace) -> None:
    for package_name in args.include_package:
        import_module_and_submodules(package_name)

    if args.user_dir is not None:
        module_name = import_user_module(args.user_dir)
        import_module_and_submodules(module_name)


def add_plugin_resolve_args(parser: DataclassArgumentParser) -> None:
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


def register_subcommands(
    subcommand_names: Iterable[Text], subparsers: _SubParsersAction
) -> None:
    for subcommand_name in subcommand_names:
        subcommand_cls = Subcommand.from_name(subcommand_name)
        subcommand = subcommand_cls()

        subparser, defaults = subcommand.add_subparser(subparsers)
        subparser.set_defaults(**defaults)
        add_plugin_resolve_args(subparser)  # duplicate args to avoid errors


def make_parser(prog: Optional[Text] = None) -> DataclassArgumentParser:
    # create an argument parser with plugin-resolving args
    parser = DataclassArgumentParser(prog=prog)
    add_plugin_resolve_args(parser)

    # create subparsers for working with subcommands
    subparsers = parser.add_subparsers(
        title="Commands", parser_class=DataclassArgumentParser
    )

    # add built-in subcommands with their configs.
    # note, that we do it prior to registering additional plugins
    # to display contextualized help messages if users ask
    default_subcommands = set(sorted(Subcommand.list_available()))
    register_subcommands(default_subcommands, subparsers)

    # register additional classes with specified packages or user-dir
    args = parser.parse_known_args()[0]
    register_additional_classes(args)

    # register user-defined custom subcommands if any
    all_subcommands = set(sorted(Subcommand.list_available()))
    user_subcommands = all_subcommands.difference(default_subcommands)
    register_subcommands(user_subcommands, subparsers)

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

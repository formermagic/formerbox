from typing import Optional, Text

from gitnetic.cli.subcommand import Subcommand
from gitnetic.common.dataclass_argparse import DataclassArgumentParser
from gitnetic.common.utils import import_module_and_submodules


def make_parser(prog: Optional[Text] = None) -> DataclassArgumentParser:
    parser = DataclassArgumentParser(prog=prog)
    subparsers = parser.add_subparsers(
        title="Commands", parser_class=DataclassArgumentParser
    )

    for subcommand_name in sorted(Subcommand.list_available()):
        subcommand_cls = Subcommand.from_name(subcommand_name)
        subcommand = subcommand_cls()

        subparser, defaults = subcommand.add_subparser(subparsers)
        subparser.set_defaults(**defaults)
        subparser.add_argument(
            "--include-package",
            type=str,
            action="append",
            default=[],
            help="additional packages to include",
        )

    return parser


def main(prog: Optional[Text] = None) -> None:
    parser = make_parser(prog)
    args = parser.parse_known_args()[0]

    # If a subparser is triggered, it adds its work as `args.func`.
    # So if no such attribute has been added, no subparser was triggered,
    # so give the user some help.
    if "func" in dir(args):
        # Import any additional modules needed (to register custom classes).
        for package_name in args.include_package:
            import_module_and_submodules(package_name)
        if hasattr(args, "add_dynamic_args"):
            args.add_dynamic_args(parser)
        params = parser.parse_args_into_dataclasses()
        args.func(params)
    else:
        parser.print_help()

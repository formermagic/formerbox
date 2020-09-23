from time import time

from gitnetic.common.dataclass_argparse import DataclassArgumentParser
from gitnetic.data.dataset_converter import DatasetConverter


def setup_parser(parent_parser: DataclassArgumentParser) -> DataclassArgumentParser:
    parser = DataclassArgumentParser(parents=[parent_parser], add_help=False)
    # fmt: off
    parser.add_argument("--converter_type", type=str, default=None, required=True,
                        help="")
    # fmt: on
    return parser


def main() -> None:
    parser = DataclassArgumentParser()
    parser = setup_parser(parser)
    args = vars(parser.parse_known_args()[0])

    converter_cls, converter_init = DatasetConverter.from_registry(
        args.pop("converter_type")
    )

    parser = converter_cls.add_argparse_args(parser)
    params = parser.parse_args_into_dataclasses()

    start = time()
    converter = converter_init(params=params[0])
    converter.convert()
    print(f"Time elapsed: {time() - start}")


if __name__ == "__main__":
    main()

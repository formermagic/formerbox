from argparse import ArgumentParser
from typing import Any, Dict, Text

from gitnetic.tasks.base_transformers import TokenizerModule


def parse_args(parent_parser: ArgumentParser) -> Dict[Text, Any]:
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    # fmt: off
    parser.add_argument("--tokenizer_type", type=str, default=None, required=True,
                        help="")
    # fmt: on

    args = vars(parser.parse_known_args()[0])
    return args


def main() -> None:
    parser = ArgumentParser()

    # parse args to build a tokenizer trainer class
    args = parse_args(parser)

    tokenizer_type = args["tokenizer_type"]
    tokenizer_cls, _ = TokenizerModule.from_registry(tokenizer_type)

    # add selected tokenizer's args
    parser = tokenizer_cls.add_argparse_args(parser, stage="train")
    args = vars(parser.parse_known_args()[0])

    tokenizer = tokenizer_cls(**args)
    tokenizer.train_tokenizer(**args)
    tokenizer.save_pretrained(**args)


if __name__ == "__main__":
    main()

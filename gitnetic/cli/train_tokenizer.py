from argparse import ArgumentParser
from typing import Any, Dict, Text

from gitnetic.tasks.base_transformers.base_tokenization import TokenizerTrainer


def parse_args(parent_parser: ArgumentParser) -> Dict[Text, Any]:
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    # fmt: off
    parser.add_argument("--tokenizer_trainer_name", type=str, default=None, required=True,
                        help="")
    parser.add_argument("--tokenizer_output_path", type=str, default=None, required=True,
                        help="")
    # fmt: on

    args = vars(parser.parse_known_args()[0])
    return args


def main() -> None:
    parser = ArgumentParser()

    # parse args to build a tokenizer trainer class
    args = parse_args(parser)
    trainer_name = args["tokenizer_trainer_name"]
    tokenizer_output_path = args["tokenizer_output_path"]
    trainer_cls, trainer_init = TokenizerTrainer.from_registry(trainer_name)

    # add selected trainer's args
    parser = trainer_cls.add_argparse_args(parser)

    # parse args for training the tokenizer
    args = vars(parser.parse_known_args()[0])
    trainer = trainer_init(args)
    trainer.train(**args)
    trainer.save_pretrained(tokenizer_output_path)


if __name__ == "__main__":
    main()

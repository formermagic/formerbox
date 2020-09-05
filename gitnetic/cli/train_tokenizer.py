from argparse import ArgumentParser
from typing import Any, Dict, Text


from gitnetic.tasks.base_transformers import TokenizerFastModule, TokenizerTrainer


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
    tokenizer_cls, _ = TokenizerFastModule.from_registry(tokenizer_type)
    trainer_cls = tokenizer_cls.trainer_cls

    # add selected tokenizer's args
    parser = tokenizer_cls.add_argparse_args(parser)
    parser = trainer_cls.add_argparse_args(parser)
    args = vars(parser.parse_known_args()[0])

    trainer = trainer_cls.from_args(tokenizer_module_cls=tokenizer_cls, **args)
    trainer.train(**args)
    trainer.save_pretrained(**args)


if __name__ == "__main__":
    main()

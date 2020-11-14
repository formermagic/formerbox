import logging
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Any, Dict, Text, Tuple, Union

from formerbox.cli.subcommand import Subcommand, _SubParsersAction
from formerbox.common.dataclass_argparse import (
    MISSING,
    DataclassArgumentParser,
    DataclassBase,
    get_params_item,
)
from formerbox.tasks import TaskModule
from formerbox.training import TransformerTrainer

logger = logging.getLogger(__name__)


@Subcommand.register("train")
class Train(Subcommand):
    @dataclass
    class Params(DataclassBase):
        task: Text = field(
            default=MISSING,
            metadata={
                "choices": TaskModule,
                "help": "The name of a registered task to perform training for.",
            },
        )

    def add_subparser(
        self, parser: _SubParsersAction
    ) -> Tuple[DataclassArgumentParser, Dict[Text, Any]]:
        description = """Train a transformer based model."""
        subparser = self._add_parser(
            parser,
            name=self.name,
            description=description,
            help=description,
        )

        # get the type of a converter to use
        subparser.add_arguments(self.Params)

        # add transformer trainer args
        TransformerTrainer.add_argparse_params(subparser)

        def add_dynamic_args(parser: DataclassArgumentParser) -> None:
            # get the parsed command arguments
            parsed_params = subparser.parse_args_into_dataclasses(
                return_remaining_strings=True
            )
            params = parsed_params[0]
            assert isinstance(params, self.Params)

            # add dybamic args to the subparser
            task_cls, _ = TaskModule.from_registry(params.task)
            task_cls.add_argparse_params(subparser)

            # inject dataclass_types to the parent parser
            parser.dataclass_types = subparser.dataclass_types

        defaults = dict(
            func=train,
            add_dynamic_args=add_dynamic_args,
        )

        return subparser, defaults


def train(params: Tuple[Union[DataclassBase, Namespace], ...]) -> None:
    cmd_params = get_params_item(params, params_type=Train.Params)
    assert cmd_params is not None

    task_cls, _ = TaskModule.from_registry(cmd_params.task)
    task_module = task_cls.setup(params=params)

    trainer_params = get_params_item(params, params_type=TransformerTrainer.Params)
    trainer_args = vars(get_params_item(params, params_type=Namespace))
    assert trainer_params is not None
    assert trainer_args is not None

    trainer = TransformerTrainer(
        task=task_module,
        params=trainer_params,
        trainer_args=trainer_args,
    )

    trainer.train()

import logging
from argparse import Namespace
from typing import Any, Dict, Text, Tuple, Union

from typeguard import typechecked

from gitnetic.cli.subcommand import Subcommand, _SubParsersAction
from gitnetic.common.dataclass_argparse import (
    DataclassArgumentParser,
    DataclassBase,
    get_params_item,
    get_parsed_attr,
)
from gitnetic.tasks.base_transformers.base_task import TaskModule
from gitnetic.tasks.base_transformers.base_trainer import TransformerTrainer

logger = logging.getLogger(__name__)


@Subcommand.register("train")
class Train(Subcommand):
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
        choices = sorted(TaskModule.list_available())
        subparser.add_argument(
            "--task",
            type=str,
            choices=choices,
            required=True,
            help="",
        )

        # add transformer trainer args
        TransformerTrainer.add_argparse_args(subparser)

        def add_dynamic_args(parser: DataclassArgumentParser) -> None:
            # add dybamic args to the subparser
            args = subparser.parse_known_args()[0]
            task_cls, _ = TaskModule.from_registry(args.task)
            task_cls.add_argparse_args(subparser)
            # inject dataclass_types to the parent parser
            parser.dataclass_types = subparser.dataclass_types

        defaults = dict(
            func=train,
            add_dynamic_args=add_dynamic_args,
        )

        return subparser, defaults


@typechecked
def train(params: Tuple[Union[DataclassBase, Namespace], ...]) -> None:
    task = get_parsed_attr(params, attribute_name="task")
    task_cls, _ = TaskModule.from_registry(task)
    task_module = task_cls.setup(params)

    trainer_params = get_params_item(params, params_type=TransformerTrainer.Params)
    trainer_args = vars(get_params_item(params, params_type=Namespace))
    trainer = TransformerTrainer(
        task=task_module,
        params=trainer_params,
        trainer_args=trainer_args,
    )

    trainer.train()

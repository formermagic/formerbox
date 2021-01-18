import logging
import os
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Text, Type

from formerbox.common.dataclass_argparse import DataclassArgumentParser, DataclassBase
from formerbox.common.has_params import HasParsableParams
from formerbox.common.registrable import Registrable
from formerbox.modules.callbacks import SaveCheckpointAtStep
from formerbox.tasks.task_module import TaskModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import Callback, EarlyStopping
from pytorch_lightning.loggers.wandb import LightningLoggerBase, WandbLogger

logger = logging.getLogger(__name__)


class TransformerTrainer(Registrable, HasParsableParams):
    @dataclass
    class Params(DataclassBase):
        wandb_project: Optional[Text] = field(
            default=None,
            metadata={"help": "The WandB project name to write logs to."},
        )
        wandb_name: Optional[Text] = field(
            default=None,
            metadata={"help": "The WandB experiment name to write logs to."},
        )
        wandb_id: Optional[Text] = field(
            default=None,
            metadata={"help": "The WandB id to use for resuming."},
        )
        save_dir: Optional[Text] = field(
            default=None,
            metadata={"help": "The dir to save training checkpoints."},
        )
        save_step_frequency: Optional[int] = field(
            default=None,
            metadata={"help": "The interval of steps between checkpoints saving."},
        )
        num_last_checkpoints: int = field(
            default=2,
            metadata={"help": "A number of last checkpoints to keep."},
        )
        seed: Optional[int] = field(
            default=None,
            metadata={"help": "A seed to make experiments reproducible."},
        )

    params: Params
    params_type: Type[Params] = Params

    def __init__(
        self,
        task: TaskModule,
        params: Params,
        trainer_args: Dict[Text, Any],
    ) -> None:
        super().__init__()
        self.task = task
        self.params = params
        self.trainer_args = trainer_args

    def train(self) -> None:
        # make a mutable copy of trainer args
        args = self.trainer_args.copy()

        # mark: setup deterministic mode
        if self.params.seed is not None:
            seed_everything(self.params.seed)
            deterministic = True
        else:
            deterministic = args.pop("deterministic", False)

        # mark: setup save checkpoint callbacks
        callbacks: List[Callback] = []
        save_dir = self.params.save_dir or os.getcwd()
        save_step_frequency = self.params.save_step_frequency
        if save_step_frequency is not None:
            num_last_checkpoints = self.params.num_last_checkpoints
            save_callback = SaveCheckpointAtStep(
                save_step_frequency, save_dir, num_last_checkpoints
            )

            callbacks.append(save_callback)

        # mark: setup loggers
        loggers: List[LightningLoggerBase] = []

        wandb_project = self.params.wandb_project
        wandb_name = self.params.wandb_name
        wandb_id = self.params.wandb_id

        wandb_required_values = [wandb_project, wandb_name]
        if all(v for v in wandb_required_values):
            wandb_logger = WandbLogger(
                project=wandb_project,
                name=wandb_name,
                id=wandb_id,
            )

            # we cannot watch the model gradients,
            # because wandb uses lambdas in backward hooks
            # which cannot be pickled by lightning
            # TODO: return watching once either of them fixes the issue
            # wandb_logger.watch(transformer_model, log_freq=1)

            loggers.append(wandb_logger)

        # mark: setup early stopping
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=0.00,
            patience=5,
            verbose=False,
            mode="min",
        )

        # mark: items to override in args
        override_kwargs: Dict[Text, Any] = {
            "replace_sampler_ddp": False,
            "reload_dataloaders_every_epoch": True,
            "callbacks": callbacks,
            "default_root_dir": save_dir,
            "checkpoint_callback": False,
            "early_stop_callback": early_stop_callback,
            "deterministic": deterministic,
            "logger": loggers,
        }

        # prepare a trainer
        trainer_args = Namespace(**{**args, **override_kwargs})
        pl_trainer: Trainer = Trainer.from_argparse_args(trainer_args)

        # run the train loop
        # pylint: disable=no-member
        pl_trainer.fit(model=self.task.module, datamodule=self.task.datamodule)

    @classmethod
    def add_argparse_params(cls, parser: DataclassArgumentParser) -> None:
        parser.add_arguments(cls.Params)
        cls.add_pl_argparse_args(parser)

    @classmethod
    def add_pl_argparse_args(cls, parser: DataclassArgumentParser) -> None:
        # pylint: disable=protected-access
        pl_parser = ArgumentParser()
        pl_parser = Trainer.add_argparse_args(pl_parser)
        for action in pl_parser._actions:
            # skip ambigious help actions
            if action.dest == "help":
                continue
            parser._add_action(action)

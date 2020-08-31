import os
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Text, Type

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import Callback, EarlyStopping
from pytorch_lightning.loggers.wandb import LightningLoggerBase, WandbLogger

from .base_task import TransformerTask
from .callbacks import SaveCheckpointAtStep


@dataclass
class TransformerTrainer:
    args: Dict[Text, Any]
    task: TransformerTask

    @classmethod
    def from_task(
        cls, task_cls: Type[TransformerTask], args: Optional[Dict[Text, Any]] = None
    ) -> "TransformerTrainer":
        if args is None:
            # prepare the arg parser
            parser = ArgumentParser()
            parser = cls.add_argparse_args(parser)
            parser = task_cls.add_argparse_args(parser)
            # parse the arguments
            args = vars(parser.parse_args())
        # setup a task instance with args
        task = task_cls.setup(args)
        return cls(args, task)

    def train(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs  # reserve for future args

        # mark: setup deterministic mode
        seed = self.args["seed"]
        deterministic = self.args.pop("deterministic", False)
        if seed is not None or deterministic:
            seed_everything(seed)

        # mark: setup save checkpoint callbacks
        callbacks: List[Callback] = []
        save_dir = self.args["save_dir"] or os.getcwd()
        save_step_frequency = self.args["save_step_frequency"]
        if save_step_frequency is not None:
            num_last_checkpoints = self.args["num_last_checkpoints"]
            save_callback = SaveCheckpointAtStep(
                save_step_frequency, save_dir, num_last_checkpoints
            )

            callbacks.append(save_callback)

        # mark: setup loggers
        loggers: List[LightningLoggerBase] = []

        wandb_project = self.args["wandb_project"]
        wandb_name = self.args["wandb_name"]
        wandb_id = self.args["wandb_id"]

        required_values = [wandb_project, wandb_name]
        if all(v for v in required_values):
            wandb_logger = WandbLogger(
                project=wandb_project,
                name=wandb_name,
                id=wandb_id,
            )

            transformer_model = self.task.module.model
            wandb_logger.watch(transformer_model, log_freq=1)

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
        trainer_args = Namespace(**{**self.args, **override_kwargs})
        pl_trainer = Trainer.from_argparse_args(trainer_args)

        # run the train loop
        pl_trainer.fit(model=self.task.module, datamodule=self.task.datamodule)

    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parent_parser = Trainer.add_argparse_args(parent_parser)
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # fmt: off
        parser.add_argument("--wandb_project", type=str, default=None, required=False,
                            help="The WandB project name to write logs to.")
        parser.add_argument("--wandb_name", type=str, default=None, required=False,
                            help="The WandB experiment name to write logs to.")
        parser.add_argument("--wandb_id", type=str, default=None, required=False,
                            help="The WandB id to use for resuming.")
        parser.add_argument("--save_dir", type=str, default=None, required=False,
                            help="The dir to save training checkpoints.")
        parser.add_argument("--save_step_frequency", type=int, default=None, required=False,
                            help="The interval of steps between checkpoints saving.")
        parser.add_argument("--num_last_checkpoints", type=int, default=2, required=False,
                            help="A number of last checkpoints to keep.")
        parser.add_argument("--seed", type=int, default=17, required=False,
                            help="A seed to make experiments reproducible.")
        # fmt: on
        return parser


if __name__ == "__main__":
    """Running on a GPU example command:

    ```
        python -m gitnetic.tasks.base_transformers.base_trainer \
            --config_path <path> \
            --tokenizer_path <path> \
            --train_data_prefix <path> \
            --val_data_prefix <path> \
            --num_workers <num> \
            --max_tokens <num> \
            --warmup_steps <num> \
            --learning_rate <num> \
            --power <num> \
            --gpus 1 \
            --num_nodes 1 \
            --distributed_backend ddp \
            --max_steps 10000
    ```
    """
    trainer = TransformerTrainer.from_task(TransformerTask)
    trainer.train()

import os
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Text, Union

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import Callback, EarlyStopping
from pytorch_lightning.loggers.wandb import LightningLoggerBase, WandbLogger
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast

from .base import TrainingParams
from .base_config import model_from_config, tokenizer_from_config
from .base_lm import TransformerDataModule, TransformerModule
from .callbacks import SaveCheckpointAtStep


def parse_args() -> Dict[Text, Any]:
    parser = ArgumentParser()
    # fmt: off
    parser.add_argument("--config_path", type=str, default=None, required=True,
                        help="")
    parser.add_argument("--tokenizer_path", type=str, default=None, required=True,
                        help="")
    parser.add_argument("--num_last_checkpoints", type=int, default=2, required=False,
                        help="")
    # fmt: on

    parser = TransformerTrainer.add_argparse_args(parser)
    parser = TransformerDataModule.add_argparse_args(parser)
    parser = TransformerModule.add_argparse_args(parser)

    return vars(parser.parse_args())


def make_tokenizer(
    args: Dict[Text, Any]
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    config_path: Text = args["config_path"]
    tokenizer_path: Text = args["tokenizer_path"]
    tokenizer = tokenizer_from_config(config_path, tokenizer_path)
    assert isinstance(tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast))
    return tokenizer


def make_datamodule(
    args: Dict[Text, Any],
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
) -> TransformerDataModule:
    train_data_prefix: Text = args["train_data_prefix"]
    val_data_prefix: Text = args["val_data_prefix"]
    num_workers: int = args["num_workers"]
    max_tokens: Optional[int] = args["max_tokens"]
    batch_size: Optional[int] = args["batch_size"]

    datamodule = TransformerDataModule(
        tokenizer=tokenizer,
        train_data_prefix=train_data_prefix,
        val_data_prefix=val_data_prefix,
        max_tokens=max_tokens,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    return datamodule


def make_model(
    args: Dict[Text, Any],
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
) -> PreTrainedModel:
    config_path: Text = args["config_path"]
    model = model_from_config(
        config_path,
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    return model


def make_module(
    args: Dict[Text, Any],
    model: PreTrainedModel,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
) -> TransformerModule:
    weight_decay: float = args["weight_decay"]
    warmup_steps: int = args["warmup_steps"]
    learning_rate: float = args["learning_rate"]
    power: float = args["power"]

    # prepare training params
    training_params = TrainingParams(
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        learning_rate=learning_rate,
        power=power,
    )

    # build a transformer lightning module model
    module = TransformerModule(model, tokenizer, training_params)

    return module


@dataclass
class TransformerTrainer:
    module: TransformerModule
    datamodule: TransformerDataModule

    def train(self, args: Dict[Text, Any]) -> None:

        # TODO: parse and inject arguments for:
        # 1) acceleration hardware setup (GPU, multi-gpu, distributed backend, etc + TPU) +
        # 2) checkpointing setup (number of steps until checkpoint, savedir, etc) +
        # 3) early stopping callbacks (e.g. stop on plateau)
        # 4) deterministic mode toggle +
        # 5) loggers setup (especially, wandb) +

        # mark: setup max training steps
        try:
            max_steps = args.pop("max_steps")
        except KeyError as err:
            raise ValueError(
                "Incorrect training command argument found."
                " Make sure you have --max_steps argument included."
            ) from err

        # mark: setup deterministic mode
        seed = args["seed"]
        deterministic = args.pop("deterministic", False)
        if seed is not None or deterministic:
            seed_everything(seed)

        # mark: setup save checkpoint callbacks
        callbacks: List[Callback] = []
        save_dir = args["save_dir"] or os.getcwd()
        save_step_frequency = args["save_step_frequency"]
        if save_step_frequency is not None:
            num_last_checkpoints = args["num_last_checkpoints"]
            save_callback = SaveCheckpointAtStep(
                save_step_frequency, save_dir, num_last_checkpoints
            )

            callbacks.append(save_callback)

        # mark: setup loggers
        loggers: List[LightningLoggerBase] = []

        wandb_project = args["wandb_project"]
        wandb_name = args["wandb_name"]
        wandb_id = args["wandb_id"]

        required_values = [wandb_project, wandb_name]
        if all(v for v in required_values):
            wandb_logger = WandbLogger(
                project=wandb_project, name=wandb_name, id=wandb_id,
            )

            wandb_logger.watch(self.module.model, log_freq=1)

            loggers.append(wandb_logger)

        # mark: setup early stopping
        early_stop_callback = EarlyStopping(
            monitor="val_loss", min_delta=0.00, patience=5, verbose=False, mode="min",
        )

        # mark: items to override in args
        override_kwargs: Dict[Text, Any] = {
            "max_steps": max_steps,
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
        traaner_args = Namespace(**{**args, **override_kwargs})
        trainer = Trainer.from_argparse_args(traaner_args)

        # run the train loop
        trainer.fit(self.module, datamodule=self.datamodule)

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
        parser.add_argument("--seed", type=int, default=17, required=False,
                        help="A seed to make experiments reproducible.")
        # fmt: on
        return parser


def train(args: Dict[Text, Any]) -> None:
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

    tokenizer = make_tokenizer(args)
    model = make_model(args, tokenizer)
    transformer_datamodule = make_datamodule(args, tokenizer)
    transformer_module = make_module(args, model, tokenizer)

    trainer = TransformerTrainer(transformer_module, transformer_datamodule)
    trainer.train(args)


if __name__ == "__main__":
    # parse the arguments
    train_args = parse_args()
    # run the training
    train(train_args)

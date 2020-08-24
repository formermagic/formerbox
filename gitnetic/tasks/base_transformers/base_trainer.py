from argparse import ArgumentParser
from typing import Any, Dict, Optional, Text, Union

from pytorch_lightning import Trainer
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
)

from .base import TrainingParams
from .base_config import model_from_config, tokenizer_from_config
from .base_lm import BaseLMDataModule, BaseLMTransformer


def parse_args() -> Dict[Text, Any]:
    parser = ArgumentParser()
    # fmt: off
    parser.add_argument("--batch_size", type=int, default=None, required=False,
                        help="")
    parser.add_argument("--max_tokens", type=int, default=None, required=False,
                        help="")
    parser.add_argument("--weight_decay", type=float, default=0.01, required=False,
                        help="")
    parser.add_argument("--warmup_steps", type=int, default=4000, required=False,
                        help="")
    parser.add_argument("--learning_rate", type=float, default=5e-4, required=True,
                        help="")
    parser.add_argument("--power", type=float, default=1.0, required=False,
                        help="")

    parser.add_argument("--config_path", type=str, default=None, required=True,
                        help="")
    parser.add_argument("--tokenizer_path", type=str, default=None, required=True,
                        help="")
    parser.add_argument("--train_data_prefix", type=str, default=None, required=True,
                        help="")
    parser.add_argument("--val_data_prefix", type=str, default=None, required=True,
                        help="")
    parser.add_argument("--num_workers", type=int, default=1, required=True,
                        help="")

    parser.add_argument("--wandb_project", type=str, default=None, required=False,
                        help="The WandB project name to write logs to.")
    parser.add_argument("--wandb_name", type=str, default=None, required=False,
                        help="The WandB experiment name to write logs to.")
    parser.add_argument("--wandb_id", type=str, default=None, required=False,
                        help="The WandB id to use for resuming.")
    parser.add_argument("--save_dir", type=str, default=None, required=False,
                        help="The dir to save training checkpoints.")
    parser.add_argument("--save_interval_updates", type=int, default=None, required=False,
                        help="The interval of steps between checkpoints saving.")
    parser.add_argument("--seed", type=int, default=None, required=False,
                        help="A seed to make experiments reproducible.")
    # fmt: on

    # add lightning trainer args
    parser = Trainer.add_argparse_args(parser)

    return vars(parser.parse_args())


def make_tokenizer(args: Dict[Text, Any]) -> PreTrainedTokenizerBase:
    config_path: Text = args["config_path"]
    tokenizer_path: Text = args["tokenizer_path"]
    tokenizer = tokenizer_from_config(config_path, tokenizer_path)
    return tokenizer


def make_datamodule(
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    args: Dict[Text, Any],
) -> BaseLMDataModule:
    train_data_prefix: Text = args["train_data_prefix"]
    val_data_prefix: Text = args["val_data_prefix"]
    num_workers: int = args["num_workers"]
    max_tokens: Optional[int] = args["max_tokens"]
    batch_size: Optional[int] = args["batch_size"]

    datamodule = BaseLMDataModule(
        tokenizer=tokenizer,
        train_data_prefix=train_data_prefix,
        val_data_prefix=val_data_prefix,
        max_tokens=max_tokens,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    return datamodule


def make_model(
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    args: Dict[Text, Any],
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
    model: PreTrainedModel,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    args: Dict[Text, Any],
) -> BaseLMTransformer:
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
    module = BaseLMTransformer(model, tokenizer, training_params)

    return module


def train(args: Dict[Text, Any]) -> None:
    # build a tokenizer
    tokenizer = make_tokenizer(args)
    assert isinstance(tokenizer, PreTrainedTokenizerFast)

    # build a data module
    datamodule = make_datamodule(tokenizer, args)

    # build a model from a config file
    model = make_model(tokenizer, args)

    # build a language model module
    transformer_module = make_module(model, tokenizer, args)

    # TODO: parse and inject arguments for:
    # 1) acceleration hardware setup (GPU, multi-gpu, distributed backend, etc + TPU)
    # 2) checkpointing setup (number of steps until checkpoint, savedir, etc)
    # 3) early stopping callbacks (e.g. stop on plateau)
    # 4) deterministic mode toggle
    # 5) loggers setup (especially, wandb)

    # prepare a trainer
    trainer = Trainer(
        max_steps=100_000,
        gpus=0,
        replace_sampler_ddp=False,
        progress_bar_refresh_rate=1,
        reload_dataloaders_every_epoch=True,
        limit_train_batches=5,
        limit_val_batches=5,
    )

    # run the train loop
    trainer.fit(transformer_module, datamodule=datamodule)


if __name__ == "__main__":
    # parse the arguments
    train_args = parse_args()
    # run the training
    train(train_args)

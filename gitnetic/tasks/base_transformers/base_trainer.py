from argparse import ArgumentParser
from typing import Any, Dict, Text

from pytorch_lightning import Trainer
from transformers import PreTrainedTokenizerFast

from gitnetic.data.indexed_dataset_setup import IndexedDatasetSetup

from .base import DataParams, TrainingParams
from .base_config import model_from_config, tokenizer_from_config
from .base_lm import BaseLMTransformer


def parse_args() -> Dict[Text, Any]:
    parser = ArgumentParser()
    # fmt: off
    parser.add_argument("--batch_size", type=int, default=None, required=False,
                        help="")
    parser.add_argument("--max_tokens", type=int, default=None, required=False,
                        help="")
    parser.add_argument("--weight_decay", type=float, default=0.01, required=False,
                        help="")
    parser.add_argument("--warmup_steps", type=int, default=None, required=False,
                        help="")
    parser.add_argument("--learning_rate", type=float, default=None, required=True,
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

    # add indexed dataset setup to parse
    IndexedDatasetSetup.add_arguments(parser)

    return vars(parser.parse_args())


if __name__ == "__main__":
    # parse the arguments
    args = parse_args()

    # prepare parsed arguments
    config_path: Text = args["config_path"]
    tokenizer_path: Text = args["tokenizer_path"]
    train_data_prefix: Text = args["train_data_prefix"]
    val_data_prefix: Text = args["val_data_prefix"]
    num_workers: int = args["num_workers"]

    # build a tokenizer from a config file
    tokenizer = tokenizer_from_config(config_path, tokenizer_path)
    assert isinstance(tokenizer, PreTrainedTokenizerFast)

    # build a model from a config file
    model = model_from_config(
        config_path,
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # prepare training params
    training_params = TrainingParams(
        batch_size=4,
        weight_decay=0.01,
        warmup_steps=4_000,
        learning_rate=5e-5,
        power=1.0,
    )

    # prepare data params
    data_params = DataParams(train_data_prefix, val_data_prefix, num_workers)

    # build a transformer lightning module model
    base_lm = BaseLMTransformer(model, tokenizer, training_params, data_params)

    # prepare a trainer
    trainer = Trainer(
        max_epochs=1, gpus=0, replace_sampler_ddp=False, progress_bar_refresh_rate=1
    )

    # run train loop
    trainer.fit(base_lm)

# Training with Tasks

## Training with CLI

```shell
export WANDB_API_KEY=<KEY>

python -m gitnetic.tasks.base_transformers.base_trainer \
    --gpus 1 \
    --num_nodes 1 \
    --distributed_backend ddp \
    --max_steps 100000 \
    --config_path $model_config \
    --tokenizer_path $tokenizer_path \
    --train_data_prefix $train_prefix_path \
    --val_data_prefix $val_prefix_path \
    --num_workers 16 \
    --max_tokens 2048 \
    --warmup_steps 1000 \
    --learning_rate 5e-5 \
    --power 1.0 \
    --save_step_frequency 1000 \
    --save_dir $save_dir \
    --val_check_interval 5000 \
    --precision 16 \
    --progress_bar_refresh_rate 20 \
    --row_log_interval 20 \
    --wandb_project test_proj --wandb_name exp-aug-26-gpu \
    --seed 17 \
    --resume_from_checkpoint $save_dir/checkpoint_last.ckpt
```

## Training with code

```python
from gitnetic.tasks import TransformerTask, TransformerTrainer


args = {
    # task args
    "config_path": "<PATH>/config.yml",
    "tokenizer_path": "<PATH>",
    # datamodule args
    "train_data_prefix": "<PATH>/train.src",
    "val_data_prefix": "<PATH>/val.src",
    "num_workers": 16,
    "max_tokens": 2048,
    # module args
    "weight_decay": 0.01,
    "warmup_steps": 4000,
    "learning_rate": 5e-4,
    "power": 1.0,
    # trainer args
    "gpus": 1,
    "num_nodes": 1,
    "distributed_backend": "ddp",
    "max_steps": 100_000,
    "precision": 16,
    "save_step_frequency": 500,
    "save_dir": "<SAVE_PATH>",
    "val_check_interval": 5000,
    "seed": 17,
    "progress_bar_refresh_rate": 20,
    "row_log_interval": 20,
    "wandb_project": "test_proj",
    "wandb_name": "exp-aug-26-gpu",
    "resume_from_checkpoint": "<SAVE_PATH>/checkpoint_last.ckpt",
    ""
}

trainer = TransformerTrainer.from_task(TransformerTask, args)
trainer.train()
```

## Training with a custom Task

```python
from argparse import ArgumentParser
from typing import Any, Dict, Text

from transformers import RobertaConfig, RobertaModel, RobertaTokenizer

from gitnetic.tasks import (
    TrainingParams,
    TransformerDataModule,
    TransformerModule,
    TransformerTask,
    TransformerTrainer,
)


class CustomTask(TransformerTask):
    @classmethod
    def setup(cls, args: Dict[Text, Any]) -> "TransformerTask":
        del args  # we hardcode everything in this example

        # prepare the pretrained tokenizer
        tokenizer_path = "<PATH>"
        tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path)
        assert isinstance(tokenizer, RobertaTokenizer)

        # prepare a model to train
        config = RobertaConfig(
            vocab_size=tokenizer.vocab_size,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        model = RobertaModel(config)

        # prepare a transformer module
        training_params = TrainingParams(
            weight_decay=0.01,
            warmup_steps=4_000,
            learning_rate=3e-4,
            power=1.0,
        )

        module = TransformerModule(model, tokenizer, training_params)

        # prepare a transformer datamodule
        datamodule = TransformerDataModule(
            tokenizer,
            train_data_prefix="<PATH>/train.src",
            val_data_prefix="<PATH>/val.src",
            max_tokens=2048,
            num_workers=16,
        )

        return cls(tokenizer, module, datamodule)

    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser) -> ArgumentParser:
        return parent_parser


###
### Setup the Trainer
###

args = {
    # trainer args
    "save_step_frequency": 500,
    "save_dir": "<SAVE_PATH>",
    "val_check_interval": 1000,
    "seed": 42,
    "limit_val_batches": 0.1,
}

trainer = TransformerTrainer.from_task(CustomTask, args)
trainer.train()
```

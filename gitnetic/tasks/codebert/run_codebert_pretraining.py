import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, List, Optional, Text, Tuple, Union

import torch
import torch.cuda as cuda
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities.memory import garbage_collection_cuda, is_oom_error
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    PreTrainedTokenizerFast,
    RobertaConfig,
    RobertaForMaskedLM,
)
from transformers.data.data_collator import DataCollatorForLanguageModeling

from gitnetic.data import IndexedDataset
from gitnetic.data.samplers import (
    BatchSampler,
    DistributedBatchSampler,
    MaxTokensBatchSampler,
)
from gitnetic.optim import get_polynomial_decay_with_warmup, weight_decay_params
from gitnetic.utils import perplexity

from .tokenization_codebert import CodeBertTokenizerFast

try:
    import torch_xla.core.xla_model as xm  # type: ignore
except ImportError:
    pass


class ValidSaveCallback(Callback):
    def __init__(self, filepath: Text, save_top_k: int = 3) -> None:
        self.filepath = filepath
        self.save_top_k = save_top_k

    @staticmethod
    def _keep_last_files(num: int, dirname: Text) -> None:
        paths = sorted(Path(dirname).iterdir(), key=os.path.getmtime)
        for path in paths[:-num]:
            os.remove(path)

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        save_filepath = os.path.join(self.filepath, "{epoch}-{step}")
        model_checkpoint = ModelCheckpoint(save_filepath, save_top_k=self.save_top_k)
        save_filepath = model_checkpoint.format_checkpoint_name(
            epoch=trainer.current_epoch,
            metrics=dict(**trainer.callback_metrics, step=trainer.global_step),
        )

        model_checkpoint.save_function = trainer.save_checkpoint
        # pylint: disable=protected-access
        model_checkpoint._save_model(save_filepath)

        # keep last `save_top_k` files
        self._keep_last_files(num=self.save_top_k, dirname=self.filepath)


class CodeBertLMPretraining(LightningModule):
    tokenizer: PreTrainedTokenizerFast
    optimizer: Optional[Optimizer] = None
    lr_scheduler: Optional[LambdaLR] = None
    trainer: Optional[Trainer] = None

    # pylint: disable=too-many-arguments, too-many-locals
    def __init__(
        self,
        num_hidden_layers: int,
        num_attention_heads: int,
        batch_size: int,
        weight_decay: float,
        warmup_steps: int,
        learning_rate: float,
        power: float,
        tokenizer_path: Text,
        tokenizer_add_prefix_space: bool,
        tokenizer_trim_offsets: bool,
        tokenizer_lowercase: bool,
        train_data_path: Text,
        val_data_path: Text,
        num_workers: int,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        self.save_hyperparameters()

        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.learning_rate = learning_rate
        self.power = power
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.num_workers = num_workers

        self.tokenizer = self._load_tokenizer(
            tokenizer_path,
            tokenizer_add_prefix_space,
            tokenizer_trim_offsets,
            tokenizer_lowercase,
        )

        self.roberta_lm = self._load_model(num_hidden_layers, num_attention_heads)

    def _load_model(
        self, num_hidden_layers: int, num_attention_heads: int
    ) -> RobertaForMaskedLM:
        config = RobertaConfig(
            vocab_size=self.tokenizer.vocab_size,
            hidden_size=768,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=3072,
            hidden_act="gelu",
            max_position_embeddings=512 + 2,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        model = RobertaForMaskedLM(config)
        model.resize_token_embeddings(len(self.tokenizer))
        return model

    def _load_tokenizer(
        self,
        tokenizer_path: Text,
        tokenizer_add_prefix_space: bool,
        tokenizer_trim_offsets: bool,
        tokenizer_lowercase: bool,
    ) -> PreTrainedTokenizerFast:
        tokenizer = CodeBertTokenizerFast.from_pretrained(
            tokenizer_path,
            add_prefix_space=tokenizer_add_prefix_space,
            trim_offsets=tokenizer_trim_offsets,
            lowercase=tokenizer_lowercase,
        )

        assert isinstance(
            tokenizer, PreTrainedTokenizerFast
        ), "Tokenizer must be a subclass of PreTrainedTokenizerFast."

        return tokenizer

    @property
    def _last_learning_rate(self) -> torch.Tensor:
        # pylint: disable=not-callable
        if self.lr_scheduler is None:
            values = torch.tensor([float("nan")])
        else:
            values = self.lr_scheduler.get_last_lr()  # type: ignore
            values = torch.tensor(values).mean()
        return values

    # pylint: disable=arguments-differ
    def forward(
        self, input_ids: torch.LongTensor, labels: torch.LongTensor, **kwargs: Any,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        self.roberta_lm.train()
        outputs = self.roberta_lm(input_ids=input_ids, labels=labels)
        loss, prediction_scores = outputs[:2]
        return loss, prediction_scores

    # pylint: disable=arguments-differ, unused-argument, not-callable
    def training_step(
        self, batch: Dict[Text, torch.Tensor], batch_idx: int
    ) -> Dict[Text, Union[torch.Tensor, Dict[Text, torch.Tensor]]]:
        # prepare logging meter values
        loss, _ = self.forward(**batch)
        train_perplexity = perplexity(loss)
        learning_rate = self._last_learning_rate
        batch_size = torch.tensor([self.batch_size])
        tensorboard_logs = {
            "train_loss": loss,
            "train_ppl": train_perplexity,
            "train_lr": learning_rate,
            "train_bz": batch_size,
        }

        return {
            "loss": loss,
            "ppl": train_perplexity,
            "lr": learning_rate,
            "log": tensorboard_logs,
        }

    # pylint: disable=arguments-differ, unused-argument
    def validation_step(
        self, batch: Dict[Text, torch.Tensor], batch_idx: int
    ) -> Dict[Text, torch.Tensor]:
        # prepare loss and ppl
        loss, _ = self.forward(**batch)
        val_perplexity = perplexity(loss)
        return {"loss": loss, "ppl": val_perplexity}

    def validation_epoch_end(
        self, outputs: List[Dict[Text, torch.Tensor]]
    ) -> Dict[Text, Dict[Text, torch.Tensor]]:
        # prepare average loss and ppl
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_perplexity = torch.stack([x["ppl"] for x in outputs]).mean()
        tensorboard_logs = {
            "val_loss": avg_loss,
            "val_ppl": avg_perplexity,
        }

        return {"log": tensorboard_logs}

    def get_progress_bar_dict(self) -> Dict[Text, Union[int, str]]:
        progress_bar_dict = super().get_progress_bar_dict()
        progress_bar_dict["lr"] = "{}".format(self.lr_scheduler.get_last_lr()[-1])  # type: ignore
        return progress_bar_dict

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[Dict]]:
        skip_list = ["bias", "LayerNorm.weight"]
        parameters = weight_decay_params(
            self.roberta_lm, weight_decay=self.weight_decay, skip_list=skip_list
        )

        optimizer = AdamW(
            parameters,  # type: ignore
            betas=(0.9, 0.98),
            eps=1e-6,
            lr=self.learning_rate,
        )

        if getattr(self.trainer, "max_steps") is None:
            t_total = self._training_steps(len(self.trainer.train_dataloader))
        else:
            t_total = self.trainer.max_steps

        scheduler = get_polynomial_decay_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=t_total,
            power=self.power,
        )

        self.optimizer = optimizer
        self.lr_scheduler = scheduler

        # called after each training steps
        step_scheduler = {
            "scheduler": scheduler,
            "interval": "step",
        }

        return [optimizer], [step_scheduler]

    def _training_steps(self, dataset_len: int) -> int:
        num_gpus = self.trainer.gpus
        if isinstance(num_gpus, list):
            num_gpus = list(num_gpus)

        batch_size = self.batch_size
        per_gpu_samples = dataset_len // (batch_size * max(1, num_gpus))
        per_gpu_samples //= self.trainer.accumulate_grad_batches
        return per_gpu_samples * self.trainer.max_epochs

    def train_dataloader(self) -> DataLoader:
        dataset = IndexedDataset(filepath_prefix=self.train_data_path)
        batch_sampler = self._batch_sampler(
            data_source=dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )

        collate_fn = DataCollatorForLanguageModeling(self.tokenizer)  # type: ignore

        data_loader = DataLoader(
            dataset,
            num_workers=self.num_workers,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
        )

        return data_loader

    def val_dataloader(self) -> DataLoader:
        dataset = IndexedDataset(filepath_prefix=self.val_data_path)
        batch_sampler = self._batch_sampler(
            data_source=dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )

        collate_fn = DataCollatorForLanguageModeling(self.tokenizer)  # type: ignore

        data_loader = DataLoader(
            dataset,
            num_workers=self.num_workers,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
        )

        return data_loader

    def _batch_sampler(
        self,
        data_source: IndexedDataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
    ) -> BatchSampler:
        batch_sampler = MaxTokensBatchSampler(
            data_source=data_source,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
        )

        if self.use_tpu:
            batch_sampler = DistributedBatchSampler(
                batch_sampler,
                num_replicas=xm.xrt_world_size(),
                rank=xm.get_ordinal(),
                shuffle=shuffle,
            )

        return batch_sampler

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser,) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # fmt: off
        parser.add_argument("--tokenizer_path", type=str, default=None,
                            help="A path to pretrained tokenizer saved files.")
        parser.add_argument("--tokenizer_add_prefix_space", type=bool, default=False,
                            help="""Whether to put a space to start the input string with or not.
                            >>>> tokenizer.decode(tokenizer.encode("Hello")) = " Hello" """)
        parser.add_argument("--tokenizer_trim_offsets", type=bool, default=True,
                            help="""Whether the post processing step should trim offsets \
                            to avoid including whitespaces.""")
        parser.add_argument("--tokenizer_lowercase", type=bool, default=True,
                            help="Whether the text should be lowercased or not.")
        parser.add_argument("--warmup_steps", type=int, default=None,
                            help="A number of warmup steps to make.")
        parser.add_argument("--weight_decay", type=float, default=None,
                            help="A weight_decay value for optimizer.")
        parser.add_argument("--power", type=float, default=1.0,
                            help="A power of learning rate decay.")
        parser.add_argument("--train_data_path", type=str, default=None,
                            help="A path to the training data file.")
        parser.add_argument("--val_data_path", type=str, default=None,
                            help="A path to the validation data file.")
        parser.add_argument("--local_rank", type=int, default=-1,
                            help="local_rank for distributed training on gpus")
        parser.add_argument("--batch_size", type=int, default=1,
                            help="Batch size value for training setup.")
        parser.add_argument("--learning_rate", type=float, default=0.001,
                            help="Train learning rate for optimizer.")
        parser.add_argument("--num_hidden_layers", type=int, default=6,
                            help="A number of transformer encoder hidden layers.")
        parser.add_argument("--num_attention_heads", type=int, default=12,
                            help="A number of self-attention heads.")
        parser.add_argument("--num_workers", type=int, default=0,
                            help="A number of workers for data loaders.")
        # fmt: on
        return parser


def main() -> None:
    parser = ArgumentParser()
    # fmt: off
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="The WandB project name to write logs to.")
    parser.add_argument("--wandb_name", type=str, default=None,
                        help="The WandB experiment name to write logs to.")
    parser.add_argument("--wandb_id", type=str, default=None,
                        help="The WandB id to use for resuming.")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="The dir to save training checkpoints.")
    parser.add_argument("--save_interval_updates", type=int, default=1,
                        help="The interval of steps between checkpoints saving.")
    parser.add_argument("--find_batch_size", default=False, action="store_true",
                        help="A flag that indicates whether we should find detect batch-size.")
    parser.add_argument("--steps_per_trial", type=int, default=10,
                        help="A number of steps to try during batch_size finding.")
    parser.add_argument("--seed", type=int, default=None,
                        help="A seed to make experiments reproducible.")
    # fmt: on

    parser = CodeBertLMPretraining.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # make experiments reproducible if needed
    if args.seed is not None:
        seed_everything(args.seed)
        determenistic = True
    else:
        determenistic = False

    code_bert_model = CodeBertLMPretraining(**vars(args))

    # use correct batch_size
    if args.find_batch_size:
        # a dummy trainer for batch_size finder
        dummy_trainer = Trainer(
            gpus=args.gpus,
            num_nodes=args.num_nodes,
            accumulate_grad_batches=args.accumulate_grad_batches,
        )
        # find fitting batch_size with binsearch
        batch_size = dummy_trainer.scale_batch_size(
            code_bert_model, mode="binsearch", steps_per_trial=args.steps_per_trial,
        )
        # clear allocated gpu memory
        cuda.empty_cache()
    else:
        batch_size = args.batch_size
    code_bert_model.batch_size = batch_size

    wandb_logger = WandbLogger(
        project=args.wandb_project, name=args.wandb_name, id=args.wandb_id,
    )
    wandb_logger.watch(code_bert_model.roberta_lm, log="gradients", log_freq=1)

    val_save = ValidSaveCallback(args.save_dir)
    trainer = Trainer(
        gpus=args.gpus,
        num_nodes=args.num_nodes,
        accumulate_grad_batches=args.accumulate_grad_batches,
        max_steps=args.max_steps,
        gradient_clip_val=args.gradient_clip_val,
        val_check_interval=args.save_interval_updates,
        logger=wandb_logger,
        callbacks=[val_save],
        auto_scale_batch_size=args.auto_scale_batch_size,
        resume_from_checkpoint=args.resume_from_checkpoint,
        deterministic=determenistic,
        row_log_interval=10,
        # train_percent_check=0.1,
        # val_percent_check=0.01,
    )

    # loop the training since we want to catch errors and repeat
    while True:
        try:
            # start training the model
            is_finished = trainer.fit(code_bert_model)
            # break the loop when training is finished
            if is_finished == 1:
                break
        except RuntimeError as exception:
            if is_oom_error(exception):
                # clear gpu memory
                garbage_collection_cuda()
                # try to decrease a batch_size so OOM hopefully resolves
                batch_size = code_bert_model.batch_size
                code_bert_model.batch_size = max(1, batch_size - 1)
            else:
                # we don't know how to treat other errors
                break


if __name__ == "__main__":
    main()

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Text, Union

from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader
from transformers import DataCollator
from transformers.data.data_collator import DataCollatorForLanguageModeling

from gitnetic.data import IndexedDataset
from gitnetic.data.samplers import (
    BatchSampler,
    DistributedBatchSampler,
    MaxTokensBatchSampler,
)

try:
    import torch_xla.core.xla_model as xm  # type: ignore
    import horovod.torch as hvd  # type: ignore
except (ModuleNotFoundError, ImportError):
    pass


@dataclass
class TrainingParams:
    batch_size: int
    weight_decay: float
    warmup_steps: int
    learning_rate: float
    power: float


@dataclass
class DataParams:
    train_data_prefix: Union[Text, Path]
    val_data_prefix: Union[Text, Path]
    num_workers: int = 0


class TrainingMixin:
    training_params: TrainingParams
    data_params: DataParams


# pylint: disable=too-many-ancestors
class BaseTrainingMixin(LightningModule, TrainingMixin):
    def __init__(self) -> None:
        super().__init__()
        self.trainer: Optional[Trainer] = None

    def get_dataloader(
        self,
        dataset: IndexedDataset,
        collator: DataCollator,
        shuffle: bool,
        drop_last: bool,
    ) -> DataLoader:

        # base collator for lanuage-modeling tasks
        collator = DataCollatorForLanguageModeling(self.tokenizer)  # type: ignore

        # prepare a batch sampler
        batch_sampler = self.batch_sampler(
            dataset=dataset,
            batch_size=self.training_params.batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
        )

        return DataLoader(
            dataset,
            num_workers=self.data_params.num_workers,
            batch_sampler=batch_sampler,
            collate_fn=collator,
        )

    def batch_sampler(
        self,
        dataset: IndexedDataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
    ) -> BatchSampler:
        batch_sampler = MaxTokensBatchSampler(
            data_source=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
        )

        if self.trainer is None:
            return batch_sampler

        if self.trainer.use_tpu:
            batch_sampler = DistributedBatchSampler(
                batch_sampler,
                num_replicas=xm.xrt_world_size(),
                rank=xm.get_ordinal(),
                shuffle=shuffle,
            )
        elif self.trainer.use_horovod:
            batch_sampler = DistributedBatchSampler(
                batch_sampler, num_replicas=hvd.size(), rank=hvd.rank(), shuffle=shuffle
            )
        else:
            world_size = {
                "ddp": self.trainer.num_nodes * self.trainer.num_processes,
                "ddp_spawn": self.trainer.num_nodes * self.trainer.num_processes,
                "ddp2": self.trainer.num_nodes,
                "ddp_cpu": self.trainer.num_processes * self.trainer.num_nodes,
            }

            if self.trainer.distributed_backend is None:
                return batch_sampler

            try:
                num_replicas = world_size[self.trainer.distributed_backend]
                rank = self.trainer.global_rank
                batch_sampler = DistributedBatchSampler(
                    batch_sampler,
                    num_replicas=num_replicas,
                    rank=rank,
                    shuffle=shuffle,
                )
            except KeyError:
                pass

        return batch_sampler

    def training_steps(self, dataset_len: int) -> int:
        assert self.trainer is not None, "Trainer must be not empty"

        if self.trainer.use_tpu:
            num_devices = self.trainer.tpu_cores
        elif self.trainer.gpus is not None:
            num_devices = self.trainer.gpus
        else:
            num_devices = 1

        if isinstance(num_devices, list):
            num_devices = len(num_devices)

        batch_size = self.training_params.batch_size
        per_gpu_samples = dataset_len // (batch_size * max(1, num_devices))
        per_gpu_samples //= self.trainer.accumulate_grad_batches

        return per_gpu_samples * self.trainer.max_epochs

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Text, Union

from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader
from transformers import DataCollator
from transformers.data.data_collator import DataCollatorForLanguageModeling

from gitnetic.data.indexed_dataset import IndexedDatasetMixin
from gitnetic.data.samplers import (
    BatchSampler,
    DistributedBatchSampler,
    UniformBatchSampler,
    UniformMaxTokensBatchSampler,
)

try:
    import torch_xla.core.xla_model as xm  # type: ignore
    import horovod.torch as hvd  # type: ignore
except (ModuleNotFoundError, ImportError):
    pass


@dataclass
class TrainingParams:
    weight_decay: float
    warmup_steps: int
    learning_rate: float
    power: float


class TrainingMixin:
    training_params: TrainingParams


# pylint: disable=too-many-ancestors
class BaseTrainingMixin(LightningModule, TrainingMixin):
    def __init__(self) -> None:
        super().__init__()
        self.trainer: Optional[Trainer] = None

    def get_dataloader(
        self,
        dataset: IndexedDatasetMixin,
        collator: DataCollator,
        batch_size: Optional[int],
        max_tokens: Optional[int],
        shuffle: bool,
        drop_last: bool,
        num_workers: int,
    ) -> DataLoader:

        # base collator for lanuage-modeling tasks
        collator = DataCollatorForLanguageModeling(self.tokenizer)  # type: ignore

        # prepare a batch sampler
        batch_sampler = self.batch_sampler(
            dataset=dataset,
            batch_size=batch_size,
            max_tokens=max_tokens,
            shuffle=shuffle,
            drop_last=drop_last,
        )

        return DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            collate_fn=collator,
        )

    def batch_sampler(
        self,
        dataset: IndexedDatasetMixin,
        max_tokens: Optional[int],
        batch_size: Optional[int],
        shuffle: bool = True,
        drop_last: bool = False,
    ) -> BatchSampler:
        if max_tokens is None and batch_size is None:
            raise ValueError(
                "Unable to prepare a batch sampler."
                " You must pass either a `batch_size`"
                " or a `max_tokens` argument."
            )

        batch_sampler: BatchSampler
        if max_tokens is not None:
            batch_sampler = UniformMaxTokensBatchSampler(
                data_source=dataset,
                max_tokens=max_tokens,
                shuffle=shuffle,
                drop_last=drop_last,
            )
        else:
            assert batch_size is not None
            batch_sampler = UniformBatchSampler(
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

    def training_steps(self, batch_nums: int, max_epochs: int) -> int:
        assert self.trainer is not None, "Trainer must be not empty"

        if self.trainer.use_tpu:
            num_devices = self.trainer.tpu_cores
        elif self.trainer.num_gpus > 0:
            num_devices = self.trainer.num_gpus
        else:
            num_devices = 1

        per_device_samples = batch_nums // max(1, num_devices)
        per_device_samples //= self.trainer.accumulate_grad_batches

        return per_device_samples * max_epochs

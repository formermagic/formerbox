from typing import Any, Optional

from formerbox.contrib import LightningTrainer
from formerbox.data.indexed_dataset import IndexedDatasetBase
from formerbox.data.samplers import (
    BatchSampler,
    DistributedBatchSampler,
    UniformBatchSampler,
    UniformMaxTokensBatchSampler,
)
from torch.utils.data import DataLoader
from transformers import DataCollator

try:
    import horovod.torch as hvd  # type: ignore
    import torch_xla.core.xla_model as xm  # type: ignore
except (ModuleNotFoundError, ImportError):
    pass


ACCUMULATE_GRAD_BATCHES_ERROR = """
    `accumulate_grad_batches` property must be of type `int`. Other types are not supported now.
    You might want to override the `training_steps(...)` method to provide support for other
    types to properly parse `accumulate_grad_batches` values.
"""

TPU_CORES_ERROR = """
    `tpu_cores` property must be of type `int`. Other types are not supported now.
    You might want to override the `training_steps(...)` method to provide support for other
    types to properly parse `tpu_cores` values.
"""


class LightningModuleMixin:
    trainer: Optional[LightningTrainer]

    def get_dataloader(
        self,
        dataset: IndexedDatasetBase,
        collator: DataCollator,
        batch_size: Optional[int],
        max_tokens: Optional[int],
        shuffle: bool,
        drop_last: bool,
        num_workers: int,
    ) -> DataLoader:
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
        dataset: IndexedDatasetBase,
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

    def training_steps(self, batch_nums: int, max_epochs: int, **kwargs: Any) -> int:
        del kwargs  # nouse
        assert self.trainer is not None, "Trainer must be not empty"

        if self.trainer.use_tpu:
            assert isinstance(self.trainer.tpu_cores, int), TPU_CORES_ERROR
            num_devices: int = self.trainer.tpu_cores
        elif self.trainer.num_gpus > 0:
            num_devices = self.trainer.num_gpus
        else:
            num_devices = 1

        assert isinstance(
            self.trainer.accumulate_grad_batches, int
        ), ACCUMULATE_GRAD_BATCHES_ERROR

        per_device_samples = batch_nums // max(1, num_devices)
        per_device_samples //= self.trainer.accumulate_grad_batches

        return per_device_samples * max_epochs

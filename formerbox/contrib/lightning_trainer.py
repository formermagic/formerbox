from typing import Dict, Iterable, List, Optional, Union

from pytorch_lightning import LightningDataModule, Trainer
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.cluster_environments.cluster_environment import (
    ClusterEnvironment,
)
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.profiler import BaseProfiler
from typing_extensions import Protocol


class LightningTrainerProperties(Protocol):
    logger: Union[LightningLoggerBase, Iterable[LightningLoggerBase], bool]
    checkpoint_callback: Union[ModelCheckpoint, bool]
    callbacks: Optional[List[Callback]]
    default_root_dir: Optional[str]
    gradient_clip_val: float
    process_position: int
    num_nodes: int
    num_processes: int
    gpus: Optional[Union[List[int], str, int]]
    auto_select_gpus: bool
    tpu_cores: Optional[Union[List[int], str, int]]
    log_gpu_memory: Optional[str]
    progress_bar_refresh_rate: int
    overfit_batches: Union[int, float]
    track_grad_norm: Union[int, float, str]
    check_val_every_n_epoch: int
    fast_dev_run: bool
    accumulate_grad_batches: Union[int, Dict[int, int], List[list]]
    max_epochs: int
    min_epochs: int
    max_steps: Optional[int]
    min_steps: Optional[int]
    limit_train_batches: Union[int, float]
    limit_val_batches: Union[int, float]
    limit_test_batches: Union[int, float]
    val_check_interval: Union[int, float]
    flush_logs_every_n_steps: int
    log_every_n_steps: int
    distributed_backend: Optional[str]
    sync_batchnorm: bool
    precision: int
    weights_summary: Optional[str]
    weights_save_path: Optional[str]
    num_sanity_val_steps: int
    truncated_bptt_steps: Optional[int]
    resume_from_checkpoint: Optional[str]
    profiler: Optional[Union[BaseProfiler, bool]]
    benchmark: bool
    deterministic: bool
    reload_dataloaders_every_epoch: bool
    auto_lr_find: Union[bool, str]
    replace_sampler_ddp: bool
    terminate_on_nan: bool
    auto_scale_batch_size: Union[str, bool]
    prepare_data_per_node: bool
    cluster_environment: Optional[ClusterEnvironment]
    amp_backend: str
    amp_level: str


class DataConnectorProperties(Protocol):
    datamodule: Optional[LightningDataModule]
    _is_data_prepared: bool


class TrainLoopProperties(Protocol):
    interrupted: bool
    should_stop: bool
    total_batch_idx: int
    batch_idx: int


class LightningTrainer(
    LightningTrainerProperties,
    DataConnectorProperties,
    TrainLoopProperties,
    Trainer,
):
    ...

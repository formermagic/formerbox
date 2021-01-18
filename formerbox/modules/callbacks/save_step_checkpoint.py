import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Text

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.core.lightning import LightningModule
from torch import Tensor

logger = logging.getLogger(__name__)


class SaveCheckpointAtStep(Callback):
    JOIN_CHAR = "-"
    NAME_LAST = "last"
    FILE_EXTENSION = ".ckpt"

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        save_step_frequency: int,
        filepath: Text,
        num_last_checkpoints: int,
        monitor: Text = "val_loss",
        prefix: Text = "checkpoint",
        save_on_epoch_end: bool = True,
    ) -> None:
        self.save_step_frequency = save_step_frequency
        self.filepath = filepath
        self.num_last_checkpoints = num_last_checkpoints
        self.monitor = monitor
        self.prefix = prefix
        self.save_on_epoch_end = save_on_epoch_end
        self.best_monitor = f"best_{self.monitor}"
        self.best_monitor_value: Optional[Tensor] = None
        self.monitor_value: Optional[Tensor] = None

    def get_metrics(self, trainer: Trainer) -> Dict[Text, Tensor]:
        # get all logged metrics
        metrics: Dict[Text, Any] = deepcopy(trainer.logger_connector.logged_metrics)
        metrics.update(trainer.logger_connector.callback_metrics)
        metrics.update(trainer.logger_connector.progress_bar_metrics)
        metrics.update({"step": trainer.global_step, "epoch": trainer.current_epoch})

        # select only scalar metrics
        scalar_metrics: Dict[Text, Tensor] = {}
        for metric, value in metrics.items():
            if isinstance(value, (int, float, bool)):
                scalar_metrics[metric] = torch.tensor(value)

        return scalar_metrics

    def on_load_checkpoint(self, checkpointed_state: Dict[Text, Any]) -> None:
        self.best_monitor_value = checkpointed_state.get(self.best_monitor)
        self.monitor_value = checkpointed_state.get(self.monitor)

    def on_save_checkpoint(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> Dict[Text, Any]:
        return {
            self.best_monitor: self.best_monitor_value,
            self.monitor: self.monitor_value,
        }

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # prepare the checkpoint save dir
        Path(self.filepath).mkdir(parents=True, exist_ok=True)

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Dict[Text, Tensor],
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        # check if we should save at current step
        if trainer.global_step % self.save_step_frequency == 0:
            self.save_checkpoint(trainer)

    def on_train_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
    ) -> None:
        # save checkpoint if needed
        if self.save_on_epoch_end:
            self.save_checkpoint(trainer)

    def save_checkpoint(self, trainer: Trainer) -> None:
        # get all logged metrics
        metrics = self.get_metrics(trainer)

        # update current monitor value
        self.monitor_value = metrics.get(self.monitor)

        # save the best checkpoint with the selected monitor
        filename = self.JOIN_CHAR.join([self.prefix, self.best_monitor])
        filename = filename + self.FILE_EXTENSION
        self.save_best_checkpoint(filename, trainer)

        # save the last checkpoint
        filename = self.JOIN_CHAR.join([self.prefix, self.NAME_LAST])
        filename = filename + self.FILE_EXTENSION
        self.save_last_checkpoint(filename, trainer)

        # save current checkpoint with a global step in the name
        filename = self.format_checkpoint_name(metrics)
        self.save_last_checkpoint(filename, trainer)

        # keep only latest `num` checkpoints + best + last
        self._keep_last_files(self.num_last_checkpoints, dirname=self.filepath)

    def save_best_checkpoint(self, filename: Text, trainer: Trainer) -> None:
        # skip saving if monitor value is not logged
        if self.monitor_value is None:
            return

        # take last saved or current value
        if self.best_monitor_value is None:
            self.best_monitor_value = self.monitor_value

        # update best value metrics
        if torch.le(self.monitor_value, self.best_monitor_value).item():
            # update and sync logged metrics with the best_monitor value
            step = trainer.global_step
            metrics = {self.best_monitor: self.monitor_value}
            trainer.logger_connector.log_metrics(metrics, {}, step=step)

            # update monitor best value and save checkpoint
            self.best_monitor_value = self.monitor_value
            ckpt_path = os.path.join(self.filepath, filename)
            trainer.save_checkpoint(ckpt_path)

    def save_last_checkpoint(self, filename: Text, trainer: Trainer) -> None:
        # skip saving if monitor value is not logged
        if self.monitor_value is None:
            return

        # update monitor best value and save checkpoint
        ckpt_path = os.path.join(self.filepath, filename)
        trainer.save_checkpoint(ckpt_path)

    def _valid_path(self, path: Path) -> bool:
        # ignore directories
        if path.is_dir():
            return False

        # ignore monitor and last checkpoints
        ignore_list = [
            self.monitor,
            self.best_monitor,
            f"{self.JOIN_CHAR}{self.NAME_LAST}",
        ]

        if any(s in path.name for s in ignore_list):
            return False

        return True

    def _keep_last_files(self, num: int, dirname: Text) -> None:
        paths = [p for p in Path(dirname).iterdir() if self._valid_path(p)]
        paths = sorted(paths, key=os.path.getmtime)
        for path in paths[:-num]:
            os.remove(path)

    def format_checkpoint_name(self, metrics: Dict[Text, Tensor]) -> Text:
        epoch = metrics.get("epoch")
        global_step = metrics.get("step")
        filename_comps = [self.prefix, f"epoch={epoch}", f"step={global_step}"]
        filename = self.JOIN_CHAR.join(filename_comps)
        filename = filename + self.FILE_EXTENSION
        return filename

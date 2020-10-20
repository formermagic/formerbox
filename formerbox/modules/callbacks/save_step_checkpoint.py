import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Text

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.core.lightning import LightningModule
from torch import Tensor

logger = logging.getLogger(__name__)


class SaveCheckpointAtStep(Callback):
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
        self.monitor_start_value = torch.tensor(0.0)

    def get_metrics(self, trainer: Trainer) -> Dict[Text, Tensor]:
        result = {}
        for metric, value in trainer.logged_metrics.items():
            if isinstance(value, (int, float, bool)):
                result[metric] = torch.tensor([value])
        return result

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        del trainer  # nouse

        # make sure the monitor buffer exists
        if not hasattr(pl_module, self.best_monitor):
            raise ValueError(
                f"Unsupported monitor ({self.monitor})."
                f" Please add `{self.best_monitor}` buffer"
                f" for your model to support the monitor."
            )

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
        del outputs, batch, batch_idx, dataloader_idx  # nouse

        # check if we should save at current step
        if trainer.global_step % self.save_step_frequency == 0:
            self.save_checkpoint(trainer, pl_module)

    def on_train_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
    ) -> None:
        del outputs  # nouse

        # save checkpoint if needed
        if self.save_on_epoch_end:
            self.save_checkpoint(trainer, pl_module)

    def update_best_monitor(
        self, monitor_value: Tensor, pl_module: LightningModule
    ) -> None:
        if hasattr(pl_module, self.best_monitor):
            setattr(pl_module, self.best_monitor, monitor_value)

    def save_checkpoint(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # get the training progress attributes
        epoch = trainer.current_epoch
        global_step = trainer.global_step

        # set the initial start value from the loaded state
        self.monitor_start_value = getattr(pl_module, self.best_monitor)

        # save the last checkpoint
        filename = f"{self.prefix}_last.ckpt"
        self.save_last_checkpoint(filename, trainer, pl_module)

        # save current checkpoint with a global step in the name
        filename = f"{self.prefix}_epoch={epoch}_global_step={global_step}.ckpt"
        self.save_last_checkpoint(filename, trainer, pl_module)

        # save the best checkpoint with the selected monitor
        filename = f"{self.prefix}_{self.best_monitor}.ckpt"
        self.save_best_checkpoint(filename, trainer, pl_module)

        # keep only latest `num` checkpoints + best + last
        self._keep_last_files(self.num_last_checkpoints, dirname=self.filepath)

    def save_best_checkpoint(
        self,
        filename: Text,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        step = trainer.global_step
        metrics = self.get_metrics(trainer)
        monitor_value: Optional[Tensor] = metrics.get(self.monitor)
        best_value: Optional[Tensor] = metrics.get(self.best_monitor)

        if monitor_value is None:
            return
        if best_value is None:
            # update and sync logged metrics with the best_monitor value
            metrics[self.best_monitor] = self.monitor_start_value
            trainer.logger_connector.log_metrics(metrics, {}, step=step)
            return

        # replace the initial value
        if best_value.item() == 0:
            best_value = monitor_value

        # update best value metrics
        if torch.le(monitor_value, best_value).item():
            # update and sync logged metrics with the best_monitor value
            metrics[self.best_monitor] = monitor_value
            trainer.logger_connector.log_metrics(metrics, {}, step=step)

            # update the best_monitor buffer
            self.update_best_monitor(monitor_value, pl_module)
            if hasattr(pl_module, self.best_monitor):
                setattr(pl_module, self.best_monitor, monitor_value)

            # save the checkpoint
            ckpt_path = os.path.join(self.filepath, filename)
            trainer.save_checkpoint(ckpt_path)

    def save_last_checkpoint(
        self, filename: Text, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        metrics = self.get_metrics(trainer)
        monitor_value: Optional[Tensor] = metrics.get(self.monitor)
        if monitor_value is None:
            return

        self.update_best_monitor(monitor_value, pl_module)
        ckpt_path = os.path.join(self.filepath, filename)
        trainer.save_checkpoint(ckpt_path)

    def _valid_path(self, path: Path) -> bool:
        # ignore directories
        if path.is_dir():
            return False
        # ignore best and last checkpoints
        fullpath = path.as_posix()
        ignore_list = ["best_", "_last"]
        if any(s in fullpath for s in ignore_list):
            return False
        return True

    def _keep_last_files(self, num: int, dirname: Text) -> None:
        paths = [p for p in Path(dirname).iterdir() if self._valid_path(p)]
        paths = sorted(paths, key=os.path.getmtime)
        for path in paths[:-num]:
            os.remove(path)

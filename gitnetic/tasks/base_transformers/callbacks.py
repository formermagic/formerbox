import os
from pathlib import Path
from typing import Dict, Optional, Text

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.core.lightning import LightningModule
from torch import Tensor


class SaveCheckpointAtStep(Callback):
    def __init__(
        self,
        save_step_frequency: int,
        filepath: Text,
        monitor: Text = "val_loss",
        prefix: Text = "checkpoint",
    ) -> None:
        self.save_step_frequency = save_step_frequency
        self.filepath = filepath
        self.monitor = monitor
        self.prefix = prefix
        self.best_monitor = f"best_{self.monitor}"
        self.monitor_start_value = torch.tensor(0.0)

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        del trainer  # nouse

        # make sure the monitor buffer exists
        if not hasattr(pl_module, self.best_monitor):
            raise ValueError(
                f"Unsupported monitor ({self.monitor})."
                f" Please add `{self.best_monitor}` buffer"
                f" for your model to support the monitor."
            )

    # pylint: disable=too-many-arguments
    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Dict[Text, Tensor],
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        del batch, batch_idx, dataloader_idx
        epoch = trainer.current_epoch
        global_step = trainer.global_step

        if global_step % self.save_step_frequency == 0:
            # keep only latest `num` checkpoints + best + last
            self._keep_last_files(num=2, dirname=self.filepath)
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

    def update_best_monitor(
        self, monitor_value: Tensor, pl_module: LightningModule
    ) -> None:
        if hasattr(pl_module, self.best_monitor):
            setattr(pl_module, self.best_monitor, monitor_value)

    def save_best_checkpoint(
        self, filename: Text, trainer: Trainer, pl_module: LightningModule,
    ) -> None:
        metrics = trainer.callback_metrics
        monitor_value: Optional[Tensor] = metrics.get(self.monitor)
        best_value: Optional[Tensor] = metrics.get(self.best_monitor)

        if monitor_value is None:
            return
        if best_value is None:
            metrics[self.best_monitor] = self.monitor_start_value
            return

        # replace the initial value
        if best_value.item() == 0:
            best_value = monitor_value

        # update best value metrics
        if monitor_value.item() <= best_value.item():
            metrics[self.best_monitor] = monitor_value
            self.update_best_monitor(monitor_value, pl_module)
            if hasattr(pl_module, self.best_monitor):
                setattr(pl_module, self.best_monitor, monitor_value)

            ckpt_path = os.path.join(self.filepath, filename)
            trainer.save_checkpoint(ckpt_path)

    def save_last_checkpoint(
        self, filename: Text, trainer: Trainer, pl_module: LightningModule,
    ) -> None:
        metrics = trainer.callback_metrics
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
            print(f"Path={path}, is_dir={path.is_dir()}")
            if path.is_dir():
                continue
            if "best" in path.name:
                continue
            os.remove(path)

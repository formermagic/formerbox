from typing import Any, Optional

import torch
from pytorch_lightning.metrics import Metric


# pylint: disable=arguments-differ
class Perplexity(Metric):
    loss: torch.Tensor

    def __init__(
        self,
        compute_on_step: bool = True,
        ddp_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
    ) -> None:
        super().__init__(compute_on_step, ddp_sync_on_step, process_group)

        self.add_state(
            name="loss",
            default=torch.tensor(0),
            dist_reduce_fx="sum",
            persistent=False,
        )

    def update(self, loss: torch.Tensor) -> None:
        self.loss = self.loss + loss

    def compute(self) -> torch.Tensor:
        return torch.exp(self.loss.mean())

from typing import Any, Optional

import torch
from pytorch_lightning.metrics import Metric
from torch.nn import CrossEntropyLoss


# pylint: disable=arguments-differ
class Perplexity(Metric):
    loss: torch.Tensor

    def __init__(
        self,
        vocab_size: int,
        compute_on_step: bool = True,
        ddp_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
    ) -> None:
        super().__init__(compute_on_step, ddp_sync_on_step, process_group)
        self.vocab_size = vocab_size
        self.criterion = CrossEntropyLoss()
        self.add_state("loss", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        logits = logits.view(-1, self.vocab_size)
        labels = labels.view(-1)
        loss = self.criterion(logits.float(), labels.long())
        self.loss = self.loss + loss

    def compute(self) -> torch.Tensor:
        return torch.exp(self.loss.mean())

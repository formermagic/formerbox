import logging
from typing import Callable, Optional

import torch
from torch import Tensor
from typing_extensions import Literal

logger = logging.getLogger(__name__)

Reduction = Literal["mean", "sum"]


# pylint: disable=abstract-method
class LabelSmoothingNLLLoss(torch.nn.Module):
    def __init__(
        self,
        label_smoothing: float,
        ignore_index: Optional[int] = -100,
        should_reduce: bool = True,
        reduction: Reduction = "mean",
    ) -> None:
        super().__init__()
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index
        self.should_reduce = should_reduce
        self.reduction = reduction
        self.reduce_op = self.get_reduce_op()

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        if targets.dim() == logits.dim() - 1:
            targets = targets.unsqueeze(-1)

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=targets)
        smooth_loss = -log_probs.sum(dim=-1, keepdim=True)

        if self.ignore_index is not None:
            pad_mask = targets == self.ignore_index
            nll_loss.masked_fill_(pad_mask, 0.0)
            smooth_loss.masked_fill_(pad_mask, 0.0)
        else:
            nll_loss = nll_loss.squeeze(-1)
            smooth_loss = smooth_loss.squeeze(-1)

        if self.should_reduce:
            nll_loss = self.reduce_op(nll_loss)
            smooth_loss = self.reduce_op(smooth_loss)

        num_classes = logits.size(-1)
        smoothing_value = self.label_smoothing / num_classes
        loss = (1.0 - self.label_smoothing) * nll_loss + smoothing_value * smooth_loss

        return loss

    def get_reduce_op(self) -> Callable[..., Tensor]:
        reduce_op: Callable[..., Tensor]
        if self.reduction == "mean":
            reduce_op = torch.mean
        elif self.reduction == "sum":
            reduce_op = torch.sum
        else:
            reduce_op = torch.mean
        return reduce_op

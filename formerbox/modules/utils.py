import logging
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from typing_extensions import Literal

logger = logging.getLogger(__name__)

Reduction = Literal["mean", "sum"]


# pylint: disable=abstract-method
class LabelSmoothingNLLLoss(nn.Module):
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
        # prepare mask for ignored label indices
        if self.ignore_index is not None:
            ignore_mask = targets == self.ignore_index
        else:
            ignore_mask = torch.zeros_like(targets)

        # all dimensions without the one for batch size
        non_batch_dims = tuple(range(1, len(targets.shape)))

        # prepare weights for loss normalization
        # shape: (batch_size,)
        weights = torch.ones_like(targets)
        weights.masked_fill_(ignore_mask, value=0.0)
        weights_batch_sum = weights.sum(dim=non_batch_dims)

        # prepare flat input tensors
        # shape: (batch_size * sequence_length, num_classes)
        logits_flat = logits.view(-1, logits.size(-1))
        # shape: (batch_size * sequence_length, num_classes)
        log_probs_flat = F.log_softmax(logits_flat, dim=-1)
        # shape: (batch_size * sequence_length, 1)
        targets_flat = targets.view(-1, 1).long()

        # prepare calculation constants
        num_classes = logits.size(-1)
        smoothing_value = self.label_smoothing / num_classes

        # fill all the correct indices with (1 - label_smoothing) value
        # shape: (batch_size * sequence_length, num_classes)
        one_hot_targets = torch.zeros_like(log_probs_flat)
        one_hot_targets = one_hot_targets.scatter_(
            dim=-1,
            index=targets_flat,
            value=(1.0 - self.label_smoothing),
        )

        # prepare smoothed nll loss values
        # shape: (batch_size * sequence_length, num_classes)
        smoothed_targets = one_hot_targets + smoothing_value
        # shape: (batch_size * sequence_length, num_classes)
        negative_log_likelihood_flat = -log_probs_flat * smoothed_targets
        # shape: (batch_size * sequence_length, 1)
        negative_log_likelihood_flat = negative_log_likelihood_flat.sum(
            dim=-1,
            keepdim=True,
        )

        # shape: (batch_size, sequence_length)
        negative_log_likelihood = negative_log_likelihood_flat.view(*targets.size())
        negative_log_likelihood.masked_fill_(ignore_mask, value=0.0)

        # prepare normalized per batch loss values
        # shape: (batch_size,)
        per_batch_loss = negative_log_likelihood.sum(non_batch_dims) / weights_batch_sum

        # finalize loss calculation
        if self.should_reduce:
            # shape: (1,)
            loss = self.reduce_op(per_batch_loss)
        else:
            # shape: (batch_size,)
            loss = per_batch_loss

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

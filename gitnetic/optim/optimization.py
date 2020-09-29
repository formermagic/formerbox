from typing import Any, Dict, Iterable, List, Text, Tuple, Union

import torch
import transformers
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

Params = Union[Iterable[Tensor], Iterable[Dict]]


def get_polynomial_decay_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    power: float = 1.0,
    last_epoch: int = -1,
) -> LambdaLR:
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        steps_passed = float(current_step - num_warmup_steps)
        steps_remaining = float(num_training_steps - num_warmup_steps)
        return max(0, (1 - steps_passed / steps_remaining) ** power)

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def weight_decay_params(
    model: torch.nn.Module, weight_decay: float, skip_list: List[Text]
) -> List[Dict[Text, Any]]:
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]


class AdamW(transformers.AdamW):
    def __init__(
        self,
        params: Params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
    ) -> None:
        # pylint: disable=too-many-arguments
        super().__init__(
            params=params,  # type: ignore
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            correct_bias=correct_bias,
        )

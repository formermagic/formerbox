"""
This type stub file was generated by pyright.
"""

import torch
from typing import Callable, Iterable, Tuple
from torch.optim import Optimizer
from .utils import logging

"""
This type stub file was generated by pyright.
"""
logger = logging.get_logger(__name__)
def get_constant_schedule(optimizer: Optimizer, last_epoch: int = ...):
    """
    Create a schedule with a constant learning rate, using the learning rate set in optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    ...

def get_constant_schedule_with_warmup(optimizer: Optimizer, num_warmup_steps: int, last_epoch: int = ...):
    """
    Create a schedule with a constant learning rate preceded by a warmup period during which the learning rate
    increases linearly between 0 and the initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    ...

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=...):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0,
    after a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    ...

def get_cosine_schedule_with_warmup(optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = ..., last_epoch: int = ...):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        num_cycles (:obj:`float`, `optional`, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    ...

def get_cosine_with_hard_restarts_schedule_with_warmup(optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: int = ..., last_epoch: int = ...):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, with several hard restarts, after a warmup period during which it increases
    linearly between 0 and the initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        num_cycles (:obj:`int`, `optional`, defaults to 1):
            The number of hard restarts to use.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    ...

def get_polynomial_decay_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, lr_end=..., power=..., last_epoch=...):
    """
    Create a schedule with a learning rate that decreases as a polynomial decay
    from the initial lr set in the optimizer to end lr defined by `lr_end`,
    after a warmup period during which it increases linearly from 0 to the
    initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        lr_end (:obj:`float`, `optional`, defaults to 1e-7):
            The end LR.
        power (:obj:`float`, `optional`, defaults to 1.0):
            Power factor.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Note: `power` defaults to 1.0 as in the fairseq implementation, which in turn is
    based on the original BERT implementation at
    https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/optimization.py#L37

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.

    """
    ...

class AdamW(Optimizer):
    """
    Implements Adam algorithm with weight decay fix as introduced in
    `Decoupled Weight Decay Regularization <https://arxiv.org/abs/1711.05101>`__.

    Parameters:
        params (:obj:`Iterable[torch.nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (:obj:`float`, `optional`, defaults to 1e-3):
            The learning rate to use.
        betas (:obj:`Tuple[float,float]`, `optional`, defaults to (0.9, 0.999)):
            Adam's betas parameters (b1, b2).
        eps (:obj:`float`, `optional`, defaults to 1e-6):
            Adam's epsilon for numerical stability.
        weight_decay (:obj:`float`, `optional`, defaults to 0):
            Decoupled weight decay to apply.
        correct_bias (:obj:`bool`, `optional`, defaults to `True`):
            Whether ot not to correct bias in Adam (for instance, in Bert TF repository they use :obj:`False`).
    """
    def __init__(self, params: Iterable[torch.nn.parameter.Parameter], lr: float = ..., betas: Tuple[float, float] = ..., eps: float = ..., weight_decay: float = ..., correct_bias: bool = ...) -> None:
        ...
    
    def step(self, closure: Callable = ...):
        """
        Performs a single optimization step.

        Arguments:
            closure (:obj:`Callable`, `optional`): A closure that reevaluates the model and returns the loss.
        """
        ...
    


class Adafactor(Optimizer):
    """
    AdaFactor pytorch implementation can be used as a drop in replacement for Adam
    original fairseq code: https://github.com/pytorch/fairseq/blob/master/fairseq/optim/adafactor.py

    Paper: `Adafactor: Adaptive Learning Rates with Sublinear Memory Cost` https://arxiv.org/abs/1804.04235
    Note that this optimizer internally adjusts the learning rate depending on the *scale_parameter*, *relative_step* and
    *warmup_init* options. To use a manual (external) learning rate schedule you should set `scale_parameter=False` and `relative_step=False`.

    Arguments:
        params (:obj:`Iterable[torch.nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (:obj:`float`, `optional`):
            The external learning rate.
        eps (:obj:`Tuple[float, float]`, `optional`, defaults to (1e-30, 1e-3)):
            Regularization constants for square gradient and parameter scale respectively
        clip_threshold (:obj:`float`, `optional`, defaults 1.0):
            Threshold of root mean square of final gradient update
        decay_rate (:obj:`float`, `optional`, defaults to -0.8):
            Coefficient used to compute running averages of square
        beta1 (:obj:`float`, `optional`):
            Coefficient used for computing running averages of gradient
        weight_decay (:obj:`float`, `optional`, defaults to 0):
            Weight decay (L2 penalty)
        scale_parameter (:obj:`bool`, `optional`, defaults to :obj:`True`):
            If True, learning rate is scaled by root mean square
        relative_step (:obj:`bool`, `optional`, defaults to :obj:`True`):
            If True, time-dependent learning rate is computed instead of external learning rate
        warmup_init (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Time-dependent learning rate computation depends on whether warm-up initialization is being used

    This implementation handles low-precision (FP16, bfloat) values, but we have not thoroughly tested.

    Recommended T5 finetuning settings:
        - Scheduled LR warm-up to fixed LR
        - disable relative updates
        - use clip threshold: https://arxiv.org/abs/2004.14546

        Example::

            Adafactor(model.parameters(), lr=1e-3, relative_step=False, warmup_init=True)

        - Alternatively, relative_step with warmup_init can be used.
        - Training without LR warmup or clip threshold is not recommended. Additional optimizer operations like
          gradient clipping should not be used alongside Adafactor.

    Usage::

        # replace AdamW with Adafactor
        optimizer = Adafactor(
            model.parameters(),
            lr=1e-3,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=None,
            weight_decay=0.0,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False
        )
    """
    def __init__(self, params, lr=..., eps=..., clip_threshold=..., decay_rate=..., beta1=..., weight_decay=..., scale_parameter=..., relative_step=..., warmup_init=...) -> None:
        ...
    
    def step(self, closure=...):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        ...
    



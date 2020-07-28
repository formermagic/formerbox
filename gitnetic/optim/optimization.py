from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


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

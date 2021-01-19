"""
This type stub file was generated by pyright.
"""

import numpy as np
from typing import Any, Dict, NamedTuple, Optional, Tuple, Union
from .tokenization_utils_base import ExplicitEnum

"""
This type stub file was generated by pyright.
"""
def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf``
    (if installed).

    Args:
        seed (:obj:`int`): The seed to set.
    """
    ...

class EvalPrediction(NamedTuple):
    """
    Evaluation output (always contains labels), to be used to compute metrics.

    Parameters:
        predictions (:obj:`np.ndarray`): Predictions of the model.
        label_ids (:obj:`np.ndarray`): Targets to be matched.
    """
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: np.ndarray
    ...


class PredictionOutput(NamedTuple):
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Optional[np.ndarray]
    metrics: Optional[Dict[str, float]]
    ...


class TrainOutput(NamedTuple):
    global_step: int
    training_loss: float
    ...


PREFIX_CHECKPOINT_DIR = "checkpoint"
class EvaluationStrategy(ExplicitEnum):
    NO = ...
    STEPS = ...
    EPOCH = ...


class BestRun(NamedTuple):
    """
    The best run found by an hyperparameter search (see :class:`~transformers.Trainer.hyperparameter_search`).

    Parameters:
        run_id (:obj:`str`):
            The id of the best run (if models were saved, the corresponding checkpoint will be in the folder ending
            with run-{run_id}).
        objective (:obj:`float`):
            The objective that was obtained for this run.
        hyperparameters (:obj:`Dict[str, Any]`):
            The hyperparameters picked to get this run.
    """
    run_id: str
    objective: float
    hyperparameters: Dict[str, Any]
    ...


def default_compute_objective(metrics: Dict[str, float]) -> float:
    """
    The default objective to maximize/minimize when doing an hyperparameter search. It is the evaluation loss if no
    metrics are provided to the :class:`~transformers.Trainer`, the sum of all metrics otherwise.

    Args:
        metrics (:obj:`Dict[str, float]`): The metrics returned by the evaluate method.

    Return:
        :obj:`float`: The objective to minimize or maximize
    """
    ...

def default_hp_space_optuna(trial) -> Dict[str, float]:
    ...

def default_hp_space_ray(trial) -> Dict[str, float]:
    ...

class HPSearchBackend(ExplicitEnum):
    OPTUNA = ...
    RAY = ...


default_hp_space = { HPSearchBackend.OPTUNA: default_hp_space_optuna,HPSearchBackend.RAY: default_hp_space_ray }

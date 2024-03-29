"""
This type stub file was generated by pyright.
"""

import os
from .trainer_utils import BestRun
from .utils import logging
from .trainer_callback import TrainerCallback

"""
Integrations with other Python libraries.
"""
logger = logging.get_logger(__name__)
_has_comet = importlib.util.find_spec("comet_ml") is not None and os.getenv("COMET_MODE", "").upper() != "DISABLED"
if _has_comet:
    ...
def is_wandb_available():
    ...

def is_comet_available():
    ...

def is_tensorboard_available():
    ...

def is_optuna_available():
    ...

def is_ray_available():
    ...

def is_ray_tune_available():
    ...

def is_azureml_available():
    ...

def is_mlflow_available():
    ...

def is_fairscale_available():
    ...

def is_deepspeed_available():
    ...

def hp_params(trial):
    ...

def default_hp_search_backend():
    ...

def run_hp_search_optuna(trainer, n_trials: int, direction: str, **kwargs) -> BestRun:
    ...

def run_hp_search_ray(trainer, n_trials: int, direction: str, **kwargs) -> BestRun:
    ...

def rewrite_logs(d):
    ...

def init_deepspeed(trainer, num_training_steps):
    """
    Init DeepSpeed, after converting any relevant Trainer's args into DeepSpeed configuration

    Args:
        trainer: Trainer object
        num_training_steps: per single gpu

    Returns: model, optimizer, lr_scheduler
    """
    ...

class TensorBoardCallback(TrainerCallback):
    """
    A :class:`~transformers.TrainerCallback` that sends the logs to `TensorBoard
    <https://www.tensorflow.org/tensorboard>`__.

    Args:
        tb_writer (:obj:`SummaryWriter`, `optional`):
            The writer to use. Will instantiate one if not set.
    """
    def __init__(self, tb_writer=...) -> None:
        ...
    
    def on_train_begin(self, args, state, control, **kwargs):
        ...
    
    def on_log(self, args, state, control, logs=..., **kwargs):
        ...
    
    def on_train_end(self, args, state, control, **kwargs):
        ...
    


class WandbCallback(TrainerCallback):
    """
    A :class:`~transformers.TrainerCallback` that sends the logs to `Weight and Biases <https://www.wandb.com/>`__.
    """
    def __init__(self) -> None:
        ...
    
    def setup(self, args, state, model, reinit, **kwargs):
        """
        Setup the optional Weights & Biases (`wandb`) integration.

        One can subclass and override this method to customize the setup if needed. Find more information `here
        <https://docs.wandb.com/huggingface>`__. You can also override the following environment variables:

        Environment:
            WANDB_LOG_MODEL (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to log model as artifact at the end of training.
            WANDB_WATCH (:obj:`str`, `optional` defaults to :obj:`"gradients"`):
                Can be :obj:`"gradients"`, :obj:`"all"` or :obj:`"false"`. Set to :obj:`"false"` to disable gradient
                logging or :obj:`"all"` to log gradients and parameters.
            WANDB_PROJECT (:obj:`str`, `optional`, defaults to :obj:`"huggingface"`):
                Set this to a custom string to store results in a different project.
            WANDB_DISABLED (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to disable wandb entirely.
        """
        ...
    
    def on_train_begin(self, args, state, control, model=..., **kwargs):
        ...
    
    def on_train_end(self, args, state, control, model=..., tokenizer=..., **kwargs):
        ...
    
    def on_log(self, args, state, control, model=..., logs=..., **kwargs):
        ...
    


class CometCallback(TrainerCallback):
    """
    A :class:`~transformers.TrainerCallback` that sends the logs to `Comet ML <https://www.comet.ml/site/>`__.
    """
    def __init__(self) -> None:
        ...
    
    def setup(self, args, state, model):
        """
        Setup the optional Comet.ml integration.

        Environment:
            COMET_MODE (:obj:`str`, `optional`):
                "OFFLINE", "ONLINE", or "DISABLED"
            COMET_PROJECT_NAME (:obj:`str`, `optional`):
                Comet.ml project name for experiments
            COMET_OFFLINE_DIRECTORY (:obj:`str`, `optional`):
                Folder to use for saving offline experiments when :obj:`COMET_MODE` is "OFFLINE"

        For a number of configurable items in the environment, see `here
        <https://www.comet.ml/docs/python-sdk/advanced/#comet-configuration-variables>`__.
        """
        ...
    
    def on_train_begin(self, args, state, control, model=..., **kwargs):
        ...
    
    def on_log(self, args, state, control, model=..., logs=..., **kwargs):
        ...
    


class AzureMLCallback(TrainerCallback):
    """
    A :class:`~transformers.TrainerCallback` that sends the logs to `AzureML
    <https://pypi.org/project/azureml-sdk/>`__.
    """
    def __init__(self, azureml_run=...) -> None:
        ...
    
    def on_init_end(self, args, state, control, **kwargs):
        ...
    
    def on_log(self, args, state, control, logs=..., **kwargs):
        ...
    


class MLflowCallback(TrainerCallback):
    """
    A :class:`~transformers.TrainerCallback` that sends the logs to `MLflow <https://www.mlflow.org/>`__.
    """
    MAX_LOG_SIZE = ...
    def __init__(self) -> None:
        ...
    
    def setup(self, args, state, model):
        """
        Setup the optional MLflow integration.

        Environment:
            HF_MLFLOW_LOG_ARTIFACTS (:obj:`str`, `optional`):
                Whether to use MLflow .log_artifact() facility to log artifacts.

                This only makes sense if logging to a remote server, e.g. s3 or GCS. If set to `True` or `1`, will copy
                whatever is in TrainerArgument's output_dir to the local or remote artifact storage. Using it without a
                remote storage will just copy the files to your artifact location.
        """
        ...
    
    def on_train_begin(self, args, state, control, model=..., **kwargs):
        ...
    
    def on_log(self, args, state, control, logs, model=..., **kwargs):
        ...
    
    def on_train_end(self, args, state, control, **kwargs):
        ...
    
    def __del__(self):
        ...
    



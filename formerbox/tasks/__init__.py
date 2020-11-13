import os

from formerbox.tasks.task_module import TaskModule
from formerbox.tasks.tokenization_base import TokenizerBase
from formerbox.tasks.tokenization_roberta import RobertaTokenizer
from formerbox.tasks.tokenization_roberta_trainer import RobertaTokenizerTrainer
from formerbox.tasks.tokenization_trainer import TokenizerTrainerBase
from formerbox.tasks.transformer_task import TransformerTask
from formerbox.utils.utils import str2bool

EXPORT_TASKS = str2bool(os.environ.get("EXPORT_TASKS", default="true"))

if EXPORT_TASKS:
    from formerbox.tasks.code import (
        CodeDatasetConverter,
        CodeRobertaTokenizer,
        CodeRobertaTokenizerTrainer,
    )

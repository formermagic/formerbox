import os

from formerbox.tasks.denoising_task import DenoisingTask
from formerbox.tasks.masked_lm_task import MaskedLMTask
from formerbox.tasks.task_module import TaskModule
from formerbox.tasks.translation_task import TranslationTask
from formerbox.tasks.word_lm_task import WordLMTask
from formerbox.utils.utils import str2bool

EXPORT_TASKS = str2bool(os.environ.get("EXPORT_TASKS", default="true"))

if EXPORT_TASKS:
    from formerbox.tasks.code import (
        CodeDatasetConverter,
        CodeRobertaTokenizer,
        CodeRobertaTokenizerTrainer,
    )

import os

from formerbox.tasks.task_module import TaskModule
from formerbox.tasks.transformer_task import TransformerTask
from formerbox.tasks.transformer_tokenization import TransformerTokenizerFast
from formerbox.tasks.transformer_tokenizer_module import TransformerTokenizerModule
from formerbox.utils.utils import str2bool

EXPORT_TASKS = str2bool(os.environ.get("EXPORT_TASKS", default="true"))

if EXPORT_TASKS:
    from formerbox.tasks.code import (
        CodeDatasetConverter,
        CodeTokenizerFast,
        CodeTokenizerModule,
    )

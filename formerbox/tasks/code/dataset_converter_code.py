import logging
from dataclasses import dataclass, field
from typing import Optional, Text, Type

from formerbox.data import DatasetConverter, DefaultDatasetConverter
from formerbox.utils.code_tokenizer import tokenize_python

Instance = Optional[Text]

logger = logging.getLogger(__name__)


@DatasetConverter.register("code", constructor="from_partial")
class CodeDatasetConverter(DefaultDatasetConverter):
    @dataclass
    class Params(DefaultDatasetConverter.Params):
        keep_comments: bool = field(
            default=False,
            metadata={
                "help": (
                    "A flag indicating whether to keep comments in code or not."
                    " Default value is false. Might be useful for finetuning."
                )
            },
        )

    params: Params
    params_type: Type[Params] = Params

    def preprocess_text(self, text: Text) -> Instance:
        # workaround to avoid disambiguation in parsing text datasets
        text = text.replace("\b", "\r").replace("\r", "\n")
        # tokenize code and turn into string again
        tokens = tokenize_python(text, keep_comments=self.params.keep_comments)
        result = " ".join(tokens)
        # invalidate empty strings, e.g. when errors occur
        if not result:
            return None
        return result

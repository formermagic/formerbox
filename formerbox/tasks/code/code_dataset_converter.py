import logging
from typing import Optional, Text

from formerbox.data import DatasetConverter, TransformerDatasetConverter
from formerbox.utils.code_tokenizer import tokenize_python

Instance = Optional[Text]

logger = logging.getLogger(__name__)


@DatasetConverter.register("code-converter", constructor="from_partial")
class CodeDatasetConverter(TransformerDatasetConverter):
    def preprocess_text(self, text: Text) -> Instance:
        # workaround to avoid disambiguation in parsing text datasets
        text = text.replace("\b", "\r")
        # tokenize code and turn into string again
        tokens = tokenize_python(text, keep_comments=True)
        result = " ".join(tokens)
        # invalidate empty strings, e.g. when errors occur
        if not result:
            return None
        return result

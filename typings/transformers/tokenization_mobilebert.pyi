"""
This type stub file was generated by pyright.
"""

from .tokenization_bert import BertTokenizer
from .utils import logging

"""Tokenization classes for MobileBERT."""
logger = logging.get_logger(__name__)
VOCAB_FILES_NAMES = { "vocab_file": "vocab.txt" }
PRETRAINED_VOCAB_FILES_MAP = { "vocab_file": { "mobilebert-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/google/mobilebert-uncased/vocab.txt" } }
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = { "mobilebert-uncased": 512 }
PRETRAINED_INIT_CONFIGURATION = {  }
class MobileBertTokenizer(BertTokenizer):
    r"""
    Construct a MobileBERT tokenizer.

    :class:`~transformers.MobileBertTokenizer is identical to :class:`~transformers.BertTokenizer` and runs end-to-end
    tokenization: punctuation splitting and wordpiece.

    Refer to superclass :class:`~transformers.BertTokenizer` for usage examples and documentation concerning
    parameters.
    """
    vocab_files_names = ...
    pretrained_vocab_files_map = ...
    max_model_input_sizes = ...
    pretrained_init_configuration = ...



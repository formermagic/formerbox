"""
This type stub file was generated by pyright.
"""

from typing import TYPE_CHECKING
from ...file_utils import _BaseLazyModule, is_sentencepiece_available, is_tf_available, is_tokenizers_available, is_torch_available
from .configuration_mbart import MBART_PRETRAINED_CONFIG_ARCHIVE_MAP, MBartConfig
from .tokenization_mbart import MBartTokenizer
from .tokenization_mbart_fast import MBartTokenizerFast
from .modeling_mbart import MBART_PRETRAINED_MODEL_ARCHIVE_LIST, MBartForConditionalGeneration, MBartForQuestionAnswering, MBartForSequenceClassification, MBartModel, MBartPreTrainedModel
from .modeling_tf_mbart import TFMBartForConditionalGeneration, TFMBartModel

_import_structure = { "configuration_mbart": ["MBART_PRETRAINED_CONFIG_ARCHIVE_MAP", "MBartConfig"] }
if is_sentencepiece_available():
    ...
if is_tokenizers_available():
    ...
if is_torch_available():
    ...
if is_tf_available():
    ...
if is_sentencepiece_available():
    ...
if is_tokenizers_available():
    ...
if is_torch_available():
    ...
if is_tf_available():
    ...

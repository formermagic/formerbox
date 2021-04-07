"""
This type stub file was generated by pyright.
"""

from typing import TYPE_CHECKING
from ...file_utils import _BaseLazyModule, is_tf_available, is_tokenizers_available, is_torch_available
from .configuration_longformer import LONGFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP, LongformerConfig
from .tokenization_longformer import LongformerTokenizer
from .tokenization_longformer_fast import LongformerTokenizerFast
from .modeling_longformer import LONGFORMER_PRETRAINED_MODEL_ARCHIVE_LIST, LongformerForMaskedLM, LongformerForMultipleChoice, LongformerForQuestionAnswering, LongformerForSequenceClassification, LongformerForTokenClassification, LongformerModel, LongformerSelfAttention
from .modeling_tf_longformer import TFLongformerForMaskedLM, TFLongformerForMultipleChoice, TFLongformerForQuestionAnswering, TFLongformerForSequenceClassification, TFLongformerForTokenClassification, TFLongformerModel, TFLongformerSelfAttention, TF_LONGFORMER_PRETRAINED_MODEL_ARCHIVE_LIST

_import_structure = { "configuration_longformer": ["LONGFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP", "LongformerConfig"],"tokenization_longformer": ["LongformerTokenizer"] }
if is_tokenizers_available():
    ...
if is_torch_available():
    ...
if is_tf_available():
    ...
if is_tokenizers_available():
    ...
if is_torch_available():
    ...
if is_tf_available():
    ...
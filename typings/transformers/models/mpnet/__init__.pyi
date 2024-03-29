"""
This type stub file was generated by pyright.
"""

from typing import TYPE_CHECKING
from ...file_utils import _BaseLazyModule, is_flax_available, is_tf_available, is_tokenizers_available, is_torch_available
from .configuration_mpnet import MPNET_PRETRAINED_CONFIG_ARCHIVE_MAP, MPNetConfig
from .tokenization_mpnet import MPNetTokenizer
from .tokenization_mpnet_fast import MPNetTokenizerFast
from .modeling_mpnet import MPNET_PRETRAINED_MODEL_ARCHIVE_LIST, MPNetForMaskedLM, MPNetForMultipleChoice, MPNetForQuestionAnswering, MPNetForSequenceClassification, MPNetForTokenClassification, MPNetLayer, MPNetModel, MPNetPreTrainedModel
from .modeling_tf_mpnet import TFMPNetEmbeddings, TFMPNetForMaskedLM, TFMPNetForMultipleChoice, TFMPNetForQuestionAnswering, TFMPNetForSequenceClassification, TFMPNetForTokenClassification, TFMPNetMainLayer, TFMPNetModel, TFMPNetPreTrainedModel, TF_MPNET_PRETRAINED_MODEL_ARCHIVE_LIST

_import_structure = { "configuration_mpnet": ["MPNET_PRETRAINED_CONFIG_ARCHIVE_MAP", "MPNetConfig"],"tokenization_mpnet": ["MPNetTokenizer"] }
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

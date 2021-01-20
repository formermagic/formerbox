"""
This type stub file was generated by pyright.
"""

from typing import TYPE_CHECKING
from ...file_utils import _BaseLazyModule, is_tf_available, is_torch_available
from .configuration_transfo_xl import TRANSFO_XL_PRETRAINED_CONFIG_ARCHIVE_MAP, TransfoXLConfig
from .tokenization_transfo_xl import TransfoXLCorpus, TransfoXLTokenizer
from .modeling_transfo_xl import AdaptiveEmbedding, TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_LIST, TransfoXLForSequenceClassification, TransfoXLLMHeadModel, TransfoXLModel, TransfoXLPreTrainedModel, load_tf_weights_in_transfo_xl
from .modeling_tf_transfo_xl import TFAdaptiveEmbedding, TFTransfoXLForSequenceClassification, TFTransfoXLLMHeadModel, TFTransfoXLMainLayer, TFTransfoXLModel, TFTransfoXLPreTrainedModel, TF_TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_LIST

_import_structure = { "configuration_transfo_xl": ["TRANSFO_XL_PRETRAINED_CONFIG_ARCHIVE_MAP", "TransfoXLConfig"],"tokenization_transfo_xl": ["TransfoXLCorpus", "TransfoXLTokenizer"] }
if is_torch_available():
    ...
if is_tf_available():
    ...
if is_torch_available():
    ...
if is_tf_available():
    ...

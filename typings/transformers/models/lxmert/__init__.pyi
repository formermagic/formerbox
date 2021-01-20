"""
This type stub file was generated by pyright.
"""

from typing import TYPE_CHECKING
from ...file_utils import _BaseLazyModule, is_tf_available, is_tokenizers_available, is_torch_available
from .configuration_lxmert import LXMERT_PRETRAINED_CONFIG_ARCHIVE_MAP, LxmertConfig
from .tokenization_lxmert import LxmertTokenizer
from .tokenization_lxmert_fast import LxmertTokenizerFast
from .modeling_lxmert import LxmertEncoder, LxmertForPreTraining, LxmertForQuestionAnswering, LxmertModel, LxmertPreTrainedModel, LxmertVisualFeatureEncoder, LxmertXLayer
from .modeling_tf_lxmert import TFLxmertForPreTraining, TFLxmertMainLayer, TFLxmertModel, TFLxmertPreTrainedModel, TFLxmertVisualFeatureEncoder, TF_LXMERT_PRETRAINED_MODEL_ARCHIVE_LIST

_import_structure = { "configuration_lxmert": ["LXMERT_PRETRAINED_CONFIG_ARCHIVE_MAP", "LxmertConfig"],"tokenization_lxmert": ["LxmertTokenizer"] }
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

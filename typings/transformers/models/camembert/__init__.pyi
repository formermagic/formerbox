"""
This type stub file was generated by pyright.
"""

from typing import TYPE_CHECKING
from ...file_utils import _BaseLazyModule, is_sentencepiece_available, is_tf_available, is_tokenizers_available, is_torch_available
from .configuration_camembert import CAMEMBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, CamembertConfig
from .tokenization_camembert import CamembertTokenizer
from .tokenization_camembert_fast import CamembertTokenizerFast
from .modeling_camembert import CAMEMBERT_PRETRAINED_MODEL_ARCHIVE_LIST, CamembertForCausalLM, CamembertForMaskedLM, CamembertForMultipleChoice, CamembertForQuestionAnswering, CamembertForSequenceClassification, CamembertForTokenClassification, CamembertModel
from .modeling_tf_camembert import TFCamembertForMaskedLM, TFCamembertForMultipleChoice, TFCamembertForQuestionAnswering, TFCamembertForSequenceClassification, TFCamembertForTokenClassification, TFCamembertModel, TF_CAMEMBERT_PRETRAINED_MODEL_ARCHIVE_LIST

_import_structure = { "configuration_camembert": ["CAMEMBERT_PRETRAINED_CONFIG_ARCHIVE_MAP", "CamembertConfig"] }
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

"""
This type stub file was generated by pyright.
"""

from typing import TYPE_CHECKING
from ...file_utils import _BaseLazyModule, is_flax_available, is_tf_available, is_tokenizers_available, is_torch_available
from .configuration_roberta import ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP, RobertaConfig
from .tokenization_roberta import RobertaTokenizer
from .tokenization_roberta_fast import RobertaTokenizerFast
from .modeling_roberta import ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST, RobertaForCausalLM, RobertaForMaskedLM, RobertaForMultipleChoice, RobertaForQuestionAnswering, RobertaForSequenceClassification, RobertaForTokenClassification, RobertaModel
from .modeling_tf_roberta import TFRobertaForMaskedLM, TFRobertaForMultipleChoice, TFRobertaForQuestionAnswering, TFRobertaForSequenceClassification, TFRobertaForTokenClassification, TFRobertaMainLayer, TFRobertaModel, TFRobertaPreTrainedModel, TF_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST
from .modeling_flax_roberta import FlaxRobertaModel

_import_structure = { "configuration_roberta": ["ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP", "RobertaConfig"],"tokenization_roberta": ["RobertaTokenizer"] }
if is_tokenizers_available():
    ...
if is_torch_available():
    ...
if is_tf_available():
    ...
if is_flax_available():
    ...
if is_tokenizers_available():
    ...
if is_torch_available():
    ...
if is_tf_available():
    ...
if is_flax_available():
    ...
"""
This type stub file was generated by pyright.
"""

from typing import TYPE_CHECKING
from ...file_utils import _BaseLazyModule, is_torch_available
from .configuration_tapas import TAPAS_PRETRAINED_CONFIG_ARCHIVE_MAP, TapasConfig
from .tokenization_tapas import TapasTokenizer
from .modeling_tapas import TAPAS_PRETRAINED_MODEL_ARCHIVE_LIST, TapasForMaskedLM, TapasForQuestionAnswering, TapasForSequenceClassification, TapasModel

_import_structure = { "configuration_tapas": ["TAPAS_PRETRAINED_CONFIG_ARCHIVE_MAP", "TapasConfig"],"tokenization_tapas": ["TapasTokenizer"] }
if is_torch_available():
    ...
if is_torch_available():
    ...
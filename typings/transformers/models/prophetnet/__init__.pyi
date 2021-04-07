"""
This type stub file was generated by pyright.
"""

from typing import TYPE_CHECKING
from ...file_utils import _BaseLazyModule, is_torch_available
from .configuration_prophetnet import PROPHETNET_PRETRAINED_CONFIG_ARCHIVE_MAP, ProphetNetConfig
from .tokenization_prophetnet import ProphetNetTokenizer
from .modeling_prophetnet import PROPHETNET_PRETRAINED_MODEL_ARCHIVE_LIST, ProphetNetDecoder, ProphetNetEncoder, ProphetNetForCausalLM, ProphetNetForConditionalGeneration, ProphetNetModel, ProphetNetPreTrainedModel

_import_structure = { "configuration_prophetnet": ["PROPHETNET_PRETRAINED_CONFIG_ARCHIVE_MAP", "ProphetNetConfig"],"tokenization_prophetnet": ["ProphetNetTokenizer"] }
if is_torch_available():
    ...
if is_torch_available():
    ...
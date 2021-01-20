"""
This type stub file was generated by pyright.
"""

from typing import TYPE_CHECKING
from ...file_utils import _BaseLazyModule, is_tf_available, is_torch_available
from .configuration_blenderbot import BLENDERBOT_PRETRAINED_CONFIG_ARCHIVE_MAP, BlenderbotConfig
from .tokenization_blenderbot import BlenderbotTokenizer
from .modeling_blenderbot import BLENDERBOT_PRETRAINED_MODEL_ARCHIVE_LIST, BlenderbotForConditionalGeneration, BlenderbotModel, BlenderbotPreTrainedModel
from .modeling_tf_blenderbot import TFBlenderbotForConditionalGeneration, TFBlenderbotModel

_import_structure = { "configuration_blenderbot": ["BLENDERBOT_PRETRAINED_CONFIG_ARCHIVE_MAP", "BlenderbotConfig"],"tokenization_blenderbot": ["BlenderbotTokenizer"] }
if is_torch_available():
    ...
if is_tf_available():
    ...
if is_torch_available():
    ...
if is_tf_available():
    ...

"""
This type stub file was generated by pyright.
"""

from typing import TYPE_CHECKING
from ...file_utils import _BaseLazyModule, is_torch_available
from .configuration_mmbt import MMBTConfig
from .modeling_mmbt import MMBTForClassification, MMBTModel, ModalEmbeddings

_import_structure = { "configuration_mmbt": ["MMBTConfig"] }
if is_torch_available():
    ...
if is_torch_available():
    ...

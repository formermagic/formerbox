"""
This type stub file was generated by pyright.
"""

import flax.linen as nn
import jax
from abc import ABC, abstractmethod
from typing import Dict
from .configuration_utils import PretrainedConfig
from .utils import logging

logger = logging.get_logger(__name__)
@jax.jit
def gelu(x):
    r"""Gaussian error linear unit activation function.

    Computes the element-wise function:

    .. math::
      \mathrm{gelu}(x) = \frac{x}{2} \left(1 + \mathrm{tanh} \left(
        \sqrt{\frac{2}{\pi}} \left(x + 0.044715 x^3 \right) \right) \right)

    We explicitly use the approximation rather than the exact formulation for
    speed. For more information, see `Gaussian Error Linear Units (GELUs)
    <https://arxiv.org/abs/1606.08415>`_, section 2.
    """
    ...

ACT2FN = { "gelu": nn.gelu,"relu": nn.relu,"swish": nn.swish,"gelu_new": gelu }
class FlaxPreTrainedModel(ABC):
    config_class = ...
    pretrained_model_archive_map = ...
    base_model_prefix = ...
    model_class = ...
    def __init__(self, config: PretrainedConfig, module: nn.Module, params: Dict, seed: int = ...) -> None:
        ...
    
    @property
    def config(self) -> PretrainedConfig:
        ...
    
    @staticmethod
    @abstractmethod
    def convert_from_pytorch(pt_state: Dict, config: PretrainedConfig) -> Dict:
        ...
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r"""
        Instantiate a pretrained Flax model from a pre-trained model configuration.
        """
        ...
    
    def save_pretrained(self, folder):
        ...
    


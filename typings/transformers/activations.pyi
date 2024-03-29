"""
This type stub file was generated by pyright.
"""

import torch
import torch.nn.functional as F
from packaging import version
from .utils import logging

logger = logging.get_logger(__name__)
def gelu_new(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """
    ...

if version.parse(torch.__version__) < version.parse("1.4"):
    gelu = _gelu_python
else:
    gelu = F.gelu
def gelu_fast(x):
    ...

if version.parse(torch.__version__) < version.parse("1.7"):
    silu = _silu_python
else:
    silu = F.silu
def mish(x):
    ...

def linear_act(x):
    ...

ACT2FN = { "relu": F.relu,"silu": silu,"swish": silu,"gelu": gelu,"tanh": torch.tanh,"gelu_new": gelu_new,"gelu_fast": gelu_fast,"mish": mish,"linear": linear_act,"sigmoid": torch.sigmoid }
def get_activation(activation_string):
    ...


"""
This type stub file was generated by pyright.
"""

from typing import List, Union
from .configuration_roberta import RobertaConfig
from .utils import logging

""" Longformer configuration """
logger = logging.get_logger(__name__)
LONGFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = { "allenai/longformer-base-4096": "https://s3.amazonaws.com/models.huggingface.co/bert/allenai/longformer-base-4096/config.json","allenai/longformer-large-4096": "https://s3.amazonaws.com/models.huggingface.co/bert/allenai/longformer-large-4096/config.json","allenai/longformer-large-4096-finetuned-triviaqa": "https://s3.amazonaws.com/models.huggingface.co/bert/allenai/longformer-large-4096-finetuned-triviaqa/config.json","allenai/longformer-base-4096-extra.pos.embd.only": "https://s3.amazonaws.com/models.huggingface.co/bert/allenai/longformer-base-4096-extra.pos.embd.only/config.json","allenai/longformer-large-4096-extra.pos.embd.only": "https://s3.amazonaws.com/models.huggingface.co/bert/allenai/longformer-large-4096-extra.pos.embd.only/config.json" }
class LongformerConfig(RobertaConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.LongformerModel` or a
    :class:`~transformers.TFLongformerModel`. It is used to instantiate a Longformer model according to the specified
    arguments, defining the model architecture.

    This is the configuration class to store the configuration of a :class:`~transformers.LongformerModel`.
    It is used to instantiate an Longformer model according to the specified arguments, defining the model
    architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of
    the RoBERTa `roberta-base <https://huggingface.co/roberta-base>`__ architecture with a sequence length 4,096.

    The :class:`~transformers.LongformerConfig` class directly inherits :class:`~transformers.RobertaConfig`.
    It reuses the same defaults. Please check the parent class for more information.

    Args:
        attention_window (:obj:`int` or :obj:`List[int]`, `optional`, defaults to 512):
            Size of an attention window around each token. If an :obj:`int`, use the same size for all layers.
            To specify a different window size for each layer, use a :obj:`List[int]` where
            ``len(attention_window) == num_hidden_layers``.

    Example::

        >>> from transformers import LongformerConfig, LongformerModel

        >>> # Initializing a Longformer configuration
        >>> configuration = LongformerConfig()

        >>> # Initializing a model from the configuration
        >>> model = LongformerModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
    """
    model_type = ...
    def __init__(self, attention_window: Union[List[int], int] = ..., sep_token_id: int = ..., **kwargs) -> None:
        ...
    



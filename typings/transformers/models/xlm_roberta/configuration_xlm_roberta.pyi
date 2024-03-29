"""
This type stub file was generated by pyright.
"""

from ...utils import logging
from ..roberta.configuration_roberta import RobertaConfig

""" XLM-RoBERTa configuration """
logger = logging.get_logger(__name__)
XLM_ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP = { "xlm-roberta-base": "https://huggingface.co/xlm-roberta-base/resolve/main/config.json","xlm-roberta-large": "https://huggingface.co/xlm-roberta-large/resolve/main/config.json","xlm-roberta-large-finetuned-conll02-dutch": "https://huggingface.co/xlm-roberta-large-finetuned-conll02-dutch/resolve/main/config.json","xlm-roberta-large-finetuned-conll02-spanish": "https://huggingface.co/xlm-roberta-large-finetuned-conll02-spanish/resolve/main/config.json","xlm-roberta-large-finetuned-conll03-english": "https://huggingface.co/xlm-roberta-large-finetuned-conll03-english/resolve/main/config.json","xlm-roberta-large-finetuned-conll03-german": "https://huggingface.co/xlm-roberta-large-finetuned-conll03-german/resolve/main/config.json" }
class XLMRobertaConfig(RobertaConfig):
    """
    This class overrides :class:`~transformers.RobertaConfig`. Please check the superclass for the appropriate
    documentation alongside usage examples.
    """
    model_type = ...



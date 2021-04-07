"""
This type stub file was generated by pyright.
"""

from ...utils import logging
from ..roberta.configuration_roberta import RobertaConfig

""" CamemBERT configuration """
logger = logging.get_logger(__name__)
CAMEMBERT_PRETRAINED_CONFIG_ARCHIVE_MAP = { "camembert-base": "https://huggingface.co/camembert-base/resolve/main/config.json","umberto-commoncrawl-cased-v1": "https://huggingface.co/Musixmatch/umberto-commoncrawl-cased-v1/resolve/main/config.json","umberto-wikipedia-uncased-v1": "https://huggingface.co/Musixmatch/umberto-wikipedia-uncased-v1/resolve/main/config.json" }
class CamembertConfig(RobertaConfig):
    """
    This class overrides :class:`~transformers.RobertaConfig`. Please check the superclass for the appropriate
    documentation alongside usage examples.
    """
    model_type = ...


"""
This type stub file was generated by pyright.
"""

from ...utils import logging
from ..roberta.tokenization_roberta_fast import RobertaTokenizerFast

logger = logging.get_logger(__name__)
vocab_url = "https://huggingface.co/roberta-large/resolve/main/vocab.json"
merges_url = "https://huggingface.co/roberta-large/resolve/main/merges.txt"
tokenizer_url = "https://huggingface.co/roberta-large/resolve/main/tokenizer.json"
_all_bart_models = ["facebook/bart-base", "facebook/bart-large", "facebook/bart-large-mnli", "facebook/bart-large-cnn", "facebook/bart-large-xsum", "yjernite/bart_eli5"]
class BartTokenizerFast(RobertaTokenizerFast):
    max_model_input_sizes = ...
    pretrained_vocab_files_map = ...
    slow_tokenizer_class = ...



"""
This type stub file was generated by pyright.
"""

from ...utils import logging
from ..roberta.tokenization_roberta import RobertaTokenizer

logger = logging.get_logger(__name__)
vocab_url = "https://huggingface.co/roberta-large/resolve/main/vocab.json"
merges_url = "https://huggingface.co/roberta-large/resolve/main/merges.txt"
_all_bart_models = ["facebook/bart-base", "facebook/bart-large", "facebook/bart-large-mnli", "facebook/bart-large-cnn", "facebook/bart-large-xsum", "yjernite/bart_eli5"]
class BartTokenizer(RobertaTokenizer):
    r"""
    Construct a BART tokenizer.

    :class:`~transformers.BartTokenizer` is identical to :class:`~transformers.RobertaTokenizer` and adds a new
    :meth:`~transformers.BartTokenizer.prepare_seq2seq_batch`

    Refer to superclass :class:`~transformers.RobertaTokenizer` for usage examples and documentation concerning the
    initialization parameters and other methods.
    """
    max_model_input_sizes = ...
    pretrained_vocab_files_map = ...


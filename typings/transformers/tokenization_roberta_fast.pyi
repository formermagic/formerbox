"""
This type stub file was generated by pyright.
"""

from typing import List, Optional
from .tokenization_gpt2_fast import GPT2TokenizerFast
from .utils import logging

"""
This type stub file was generated by pyright.
"""
logger = logging.get_logger(__name__)
VOCAB_FILES_NAMES = { "vocab_file": "vocab.json","merges_file": "merges.txt","tokenizer_file": "tokenizer.json" }
PRETRAINED_VOCAB_FILES_MAP = { "vocab_file": { "roberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-vocab.json","roberta-large": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-vocab.json","roberta-large-mnli": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-vocab.json","distilroberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/distilroberta-base-vocab.json","roberta-base-openai-detector": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-vocab.json","roberta-large-openai-detector": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-vocab.json" },"merges_file": { "roberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-merges.txt","roberta-large": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-merges.txt","roberta-large-mnli": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-merges.txt","distilroberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/distilroberta-base-merges.txt","roberta-base-openai-detector": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-merges.txt","roberta-large-openai-detector": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-merges.txt" },"tokenizer_file": { "roberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-tokenizer.json","roberta-large": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-tokenizer.json","roberta-large-mnli": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-tokenizer.json","distilroberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/distilroberta-base-tokenizer.json","roberta-base-openai-detector": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-tokenizer.json","roberta-large-openai-detector": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-tokenizer.json" } }
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = { "roberta-base": 512,"roberta-large": 512,"roberta-large-mnli": 512,"distilroberta-base": 512,"roberta-base-openai-detector": 512,"roberta-large-openai-detector": 512 }
class RobertaTokenizerFast(GPT2TokenizerFast):
    """
    Construct a "fast" RoBERTa tokenizer (backed by HuggingFace's `tokenizers` library), derived from the GPT-2
    tokenizer, using byte-level Byte-Pair-Encoding.

    This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will
    be encoded differently whether it is at the beginning of the sentence (without space) or not:

    ::

        >>> from transformers import RobertaTokenizerFast
        >>> tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        >>> tokenizer("Hello world")['input_ids']
        [0, 31414, 232, 328, 2]
        >>> tokenizer(" Hello world")['input_ids']
        [0, 20920, 232, 2]

    You can get around that behavior by passing ``add_prefix_space=True`` when instantiating this tokenizer or when you
    call it on some text, but since the model was not pretrained this way, it might yield a decrease in performance.

    .. note::

        When used with ``is_split_into_words=True``, this tokenizer needs to be instantiated with
        ``add_prefix_space=True``.

    This tokenizer inherits from :class:`~transformers.PreTrainedTokenizerFast` which contains most of the main
    methods. Users should refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (:obj:`str`):
            Path to the vocabulary file.
        merges_file (:obj:`str`):
            Path to the merges file.
        errors (:obj:`str`, `optional`, defaults to :obj:`"replace"`):
            Paradigm to follow when decoding bytes to UTF-8. See `bytes.decode
            <https://docs.python.org/3/library/stdtypes.html#bytes.decode>`__ for more information.
        bos_token (:obj:`str`, `optional`, defaults to :obj:`"<s>"`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.

            .. note::

                When building a sequence using special tokens, this is not the token that is used for the beginning
                of sequence. The token used is the :obj:`cls_token`.
        eos_token (:obj:`str`, `optional`, defaults to :obj:`"</s>"`):
            The end of sequence token.

            .. note::

                When building a sequence using special tokens, this is not the token that is used for the end
                of sequence. The token used is the :obj:`sep_token`.
        sep_token (:obj:`str`, `optional`, defaults to :obj:`"</s>"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences
            for sequence classification or for a text and a question for question answering.
            It is also used as the last token of a sequence built with special tokens.
        cls_token (:obj:`str`, `optional`, defaults to :obj:`"<s>"`):
            The classifier token which is used when doing sequence classification (classification of the whole
            sequence instead of per-token classification). It is the first token of the sequence when built with
            special tokens.
        unk_token (:obj:`str`, `optional`, defaults to :obj:`"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (:obj:`str`, `optional`, defaults to :obj:`"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        mask_token (:obj:`str`, `optional`, defaults to :obj:`"<mask>"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        add_prefix_space (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to add an initial space to the input. This allows to treat the leading word just as any
            other word. (RoBERTa tokenizer detect beginning of words by the preceding space).
        trim_offsets (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether the post processing step should trim offsets to avoid including whitespaces.
    """
    vocab_files_names = ...
    pretrained_vocab_files_map = ...
    max_model_input_sizes = ...
    model_input_names = ...
    slow_tokenizer_class = ...
    def __init__(self, vocab_file, merges_file, tokenizer_file=..., errors=..., bos_token=..., eos_token=..., sep_token=..., cls_token=..., unk_token=..., pad_token=..., mask_token=..., add_prefix_space=..., **kwargs) -> None:
        ...
    
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=...):
        ...
    
    def create_token_type_ids_from_sequences(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = ...) -> List[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task.
        RoBERTa does not make use of token type ids, therefore a list of zeros is returned.

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.

        Returns:
            :obj:`List[int]`:  List of zeros.
        """
        ...
    



"""
This type stub file was generated by pyright.
"""

from typing import List, Optional, Tuple
from ...file_utils import is_sentencepiece_available
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import logging
from .tokenization_albert import AlbertTokenizer

""" Tokenization classes for ALBERT model."""
if is_sentencepiece_available():
    ...
else:
    AlbertTokenizer = None
logger = logging.get_logger(__name__)
VOCAB_FILES_NAMES = { "vocab_file": "spiece.model","tokenizer_file": "tokenizer.json" }
PRETRAINED_VOCAB_FILES_MAP = { "vocab_file": { "albert-base-v1": "https://huggingface.co/albert-base-v1/resolve/main/spiece.model","albert-large-v1": "https://huggingface.co/albert-large-v1/resolve/main/spiece.model","albert-xlarge-v1": "https://huggingface.co/albert-xlarge-v1/resolve/main/spiece.model","albert-xxlarge-v1": "https://huggingface.co/albert-xxlarge-v1/resolve/main/spiece.model","albert-base-v2": "https://huggingface.co/albert-base-v2/resolve/main/spiece.model","albert-large-v2": "https://huggingface.co/albert-large-v2/resolve/main/spiece.model","albert-xlarge-v2": "https://huggingface.co/albert-xlarge-v2/resolve/main/spiece.model","albert-xxlarge-v2": "https://huggingface.co/albert-xxlarge-v2/resolve/main/spiece.model" },"tokenizer_file": { "albert-base-v1": "https://huggingface.co/albert-base-v1/resolve/main/tokenizer.json","albert-large-v1": "https://huggingface.co/albert-large-v1/resolve/main/tokenizer.json","albert-xlarge-v1": "https://huggingface.co/albert-xlarge-v1/resolve/main/tokenizer.json","albert-xxlarge-v1": "https://huggingface.co/albert-xxlarge-v1/resolve/main/tokenizer.json","albert-base-v2": "https://huggingface.co/albert-base-v2/resolve/main/tokenizer.json","albert-large-v2": "https://huggingface.co/albert-large-v2/resolve/main/tokenizer.json","albert-xlarge-v2": "https://huggingface.co/albert-xlarge-v2/resolve/main/tokenizer.json","albert-xxlarge-v2": "https://huggingface.co/albert-xxlarge-v2/resolve/main/tokenizer.json" } }
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = { "albert-base-v1": 512,"albert-large-v1": 512,"albert-xlarge-v1": 512,"albert-xxlarge-v1": 512,"albert-base-v2": 512,"albert-large-v2": 512,"albert-xlarge-v2": 512,"albert-xxlarge-v2": 512 }
SPIECE_UNDERLINE = "▁"
class AlbertTokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a "fast" ALBERT tokenizer (backed by HuggingFace's `tokenizers` library). Based on `Unigram
    <https://huggingface.co/docs/tokenizers/python/latest/components.html?highlight=unigram#models>`__. This tokenizer
    inherits from :class:`~transformers.PreTrainedTokenizerFast` which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods

    Args:
        vocab_file (:obj:`str`):
            `SentencePiece <https://github.com/google/sentencepiece>`__ file (generally has a `.spm` extension) that
            contains the vocabulary necessary to instantiate a tokenizer.
        do_lower_case (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to lowercase the input when tokenizing.
        remove_space (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to strip the text when tokenizing (removing excess spaces before and after the string).
        keep_accents (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to keep accents when tokenizing.
        bos_token (:obj:`str`, `optional`, defaults to :obj:`"[CLS]"`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.
            .. note:: When building a sequence using special tokens, this is not the token that is used for the
            beginning of sequence. The token used is the :obj:`cls_token`.
        eos_token (:obj:`str`, `optional`, defaults to :obj:`"[SEP]"`):
            The end of sequence token. .. note:: When building a sequence using special tokens, this is not the token
            that is used for the end of sequence. The token used is the :obj:`sep_token`.
        unk_token (:obj:`str`, `optional`, defaults to :obj:`"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        sep_token (:obj:`str`, `optional`, defaults to :obj:`"[SEP]"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        pad_token (:obj:`str`, `optional`, defaults to :obj:`"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        cls_token (:obj:`str`, `optional`, defaults to :obj:`"[CLS]"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        mask_token (:obj:`str`, `optional`, defaults to :obj:`"[MASK]"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict. Attributes:
        sp_model (:obj:`SentencePieceProcessor`):
            The `SentencePiece` processor that is used for every conversion (string, tokens and IDs).
    """
    vocab_files_names = ...
    pretrained_vocab_files_map = ...
    max_model_input_sizes = ...
    slow_tokenizer_class = ...
    def __init__(self, vocab_file, tokenizer_file=..., do_lower_case=..., remove_space=..., keep_accents=..., bos_token=..., eos_token=..., unk_token=..., sep_token=..., pad_token=..., cls_token=..., mask_token=..., **kwargs) -> None:
        ...
    
    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = ...) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. An ALBERT sequence has the following format:

        - single sequence: ``[CLS] X [SEP]``
        - pair of sequences: ``[CLS] A [SEP] B [SEP]``

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added
            token_ids_1 (:obj:`List[int]`, `optional`, defaults to :obj:`None`):
                Optional second list of IDs for sequence pairs.

        Returns:
            :obj:`List[int]`: list of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
        """
        ...
    
    def get_special_tokens_mask(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = ..., already_has_special_tokens: bool = ...) -> List[int]:
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``prepare_for_model`` method.

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of ids.
            token_ids_1 (:obj:`List[int]`, `optional`, defaults to :obj:`None`):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Set to True if the token list is already formatted with special tokens for the model

        Returns:
            :obj:`List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        ...
    
    def create_token_type_ids_from_sequences(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = ...) -> List[int]:
        """
        Creates a mask from the two sequences passed to be used in a sequence-pair classification task. An ALBERT
        sequence pair mask has the following format:

        ::

            0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
            | first sequence    | second sequence |

        if token_ids_1 is None, only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of ids.
            token_ids_1 (:obj:`List[int]`, `optional`, defaults to :obj:`None`):
                Optional second list of IDs for sequence pairs.

        Returns:
            :obj:`List[int]`: List of `token type IDs <../glossary.html#token-type-ids>`_ according to the given
            sequence(s).
        """
        ...
    
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = ...) -> Tuple[str]:
        ...
    



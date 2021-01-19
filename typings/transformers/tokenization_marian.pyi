"""
This type stub file was generated by pyright.
"""

import sentencepiece
from typing import Dict, List, Optional, Tuple, Union
from .file_utils import add_start_docstrings
from .tokenization_utils import BatchEncoding, PreTrainedTokenizer
from .tokenization_utils_base import PREPARE_SEQ2SEQ_BATCH_DOCSTRING

"""
This type stub file was generated by pyright.
"""
vocab_files_names = { "source_spm": "source.spm","target_spm": "target.spm","vocab": "vocab.json","tokenizer_config_file": "tokenizer_config.json" }
PRETRAINED_VOCAB_FILES_MAP = { "source_spm": { "Helsinki-NLP/opus-mt-en-de": "https://cdn.huggingface.co/Helsinki-NLP/opus-mt-en-de/source.spm" },"target_spm": { "Helsinki-NLP/opus-mt-en-de": "https://cdn.huggingface.co/Helsinki-NLP/opus-mt-en-de/target.spm" },"vocab": { "Helsinki-NLP/opus-mt-en-de": "https://cdn.huggingface.co/Helsinki-NLP/opus-mt-en-de/vocab.json" },"tokenizer_config_file": { "Helsinki-NLP/opus-mt-en-de": "https://cdn.huggingface.co/Helsinki-NLP/opus-mt-en-de/tokenizer_config.json" } }
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = { "Helsinki-NLP/opus-mt-en-de": 512 }
PRETRAINED_INIT_CONFIGURATION = {  }
class MarianTokenizer(PreTrainedTokenizer):
    r"""
    Construct a Marian tokenizer. Based on `SentencePiece <https://github.com/google/sentencepiece>`__.

    This tokenizer inherits from :class:`~transformers.PreTrainedTokenizer` which contains most of the main methods.
    Users should refer to this superclass for more information regarding those methods.

    Args:
        source_spm (:obj:`str`):
            `SentencePiece <https://github.com/google/sentencepiece>`__ file (generally has a .spm extension) that
            contains the vocabulary for the source language.
        target_spm (:obj:`str`):
            `SentencePiece <https://github.com/google/sentencepiece>`__ file (generally has a .spm extension) that
            contains the vocabulary for the target language.
        source_lang (:obj:`str`, `optional`):
            A string representing the source language.
        target_lang (:obj:`str`, `optional`):
            A string representing the target language.
        unk_token (:obj:`str`, `optional`, defaults to :obj:`"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        eos_token (:obj:`str`, `optional`, defaults to :obj:`"</s>"`):
            The end of sequence token.
        pad_token (:obj:`str`, `optional`, defaults to :obj:`"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        model_max_length (:obj:`int`, `optional`, defaults to 512):
            The maximum sentence length the model accepts.
        additional_special_tokens (:obj:`List[str]`, `optional`, defaults to :obj:`["<eop>", "<eod>"]`):
            Additional special tokens used by the tokenizer.

    Examples::

        >>> from transformers import MarianTokenizer
        >>> tok = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-de')
        >>> src_texts = [ "I am a small frog.", "Tom asked his teacher for advice."]
        >>> tgt_texts = ["Ich bin ein kleiner Frosch.", "Tom bat seinen Lehrer um Rat."]  # optional
        >>> batch_enc: BatchEncoding = tok.prepare_seq2seq_batch(src_texts, tgt_texts=tgt_texts)
        >>> # keys  [input_ids, attention_mask, labels].
        >>> # model(**batch) should work
    """
    vocab_files_names = ...
    pretrained_vocab_files_map = ...
    pretrained_init_configuration = ...
    max_model_input_sizes = ...
    model_input_names = ...
    language_code_re = ...
    def __init__(self, vocab, source_spm, target_spm, source_lang=..., target_lang=..., unk_token=..., eos_token=..., pad_token=..., model_max_length=..., **kwargs) -> None:
        ...
    
    def normalize(self, x: str) -> str:
        """Cover moses empty string edge case. They return empty list for '' input!"""
        ...
    
    def remove_language_code(self, text: str):
        """Remove language codes like <<fr>> before sentencepiece"""
        ...
    
    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Uses target language sentencepiece model"""
        ...
    
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=...) -> List[int]:
        """Build model inputs from a sequence by appending eos_token_id."""
        ...
    
    @add_start_docstrings(PREPARE_SEQ2SEQ_BATCH_DOCSTRING)
    def prepare_seq2seq_batch(self, src_texts: List[str], tgt_texts: Optional[List[str]] = ..., max_length: Optional[int] = ..., max_target_length: Optional[int] = ..., return_tensors: str = ..., truncation=..., padding=..., **unused) -> BatchEncoding:
        ...
    
    @property
    def vocab_size(self) -> int:
        ...
    
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = ...) -> Tuple[str]:
        ...
    
    def get_vocab(self) -> Dict:
        ...
    
    def __getstate__(self) -> Dict:
        ...
    
    def __setstate__(self, d: Dict) -> None:
        ...
    
    def num_special_tokens_to_add(self, **unused):
        """Just EOS"""
        ...
    
    def get_special_tokens_mask(self, token_ids_0: List, token_ids_1: Optional[List] = ..., already_has_special_tokens: bool = ...) -> List[int]:
        """Get list where entries are [1] if a token is [eos] or [pad] else 0."""
        ...
    


def load_spm(path: str) -> sentencepiece.SentencePieceProcessor:
    ...

def save_json(data, path: str) -> None:
    ...

def load_json(path: str) -> Union[Dict, List]:
    ...


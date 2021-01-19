"""
This type stub file was generated by pyright.
"""

from typing import Optional
from .tokenization_bert import BertTokenizer
from .utils import logging

"""
This type stub file was generated by pyright.
"""
logger = logging.get_logger(__name__)
VOCAB_FILES_NAMES = { "vocab_file": "vocab.txt" }
PRETRAINED_VOCAB_FILES_MAP = { "vocab_file": { "cl-tohoku/bert-base-japanese": "https://s3.amazonaws.com/models.huggingface.co/bert/cl-tohoku/bert-base-japanese/vocab.txt","cl-tohoku/bert-base-japanese-whole-word-masking": "https://s3.amazonaws.com/models.huggingface.co/bert/cl-tohoku/bert-base-japanese-whole-word-masking/vocab.txt","cl-tohoku/bert-base-japanese-char": "https://s3.amazonaws.com/models.huggingface.co/bert/cl-tohoku/bert-base-japanese-char/vocab.txt","cl-tohoku/bert-base-japanese-char-whole-word-masking": "https://s3.amazonaws.com/models.huggingface.co/bert/cl-tohoku/bert-base-japanese-char-whole-word-masking/vocab.txt" } }
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = { "cl-tohoku/bert-base-japanese": 512,"cl-tohoku/bert-base-japanese-whole-word-masking": 512,"cl-tohoku/bert-base-japanese-char": 512,"cl-tohoku/bert-base-japanese-char-whole-word-masking": 512 }
PRETRAINED_INIT_CONFIGURATION = { "cl-tohoku/bert-base-japanese": { "do_lower_case": False,"word_tokenizer_type": "mecab","subword_tokenizer_type": "wordpiece" },"cl-tohoku/bert-base-japanese-whole-word-masking": { "do_lower_case": False,"word_tokenizer_type": "mecab","subword_tokenizer_type": "wordpiece" },"cl-tohoku/bert-base-japanese-char": { "do_lower_case": False,"word_tokenizer_type": "mecab","subword_tokenizer_type": "character" },"cl-tohoku/bert-base-japanese-char-whole-word-masking": { "do_lower_case": False,"word_tokenizer_type": "mecab","subword_tokenizer_type": "character" } }
class BertJapaneseTokenizer(BertTokenizer):
    """BERT tokenizer for Japanese text"""
    vocab_files_names = ...
    pretrained_vocab_files_map = ...
    pretrained_init_configuration = ...
    max_model_input_sizes = ...
    def __init__(self, vocab_file, do_lower_case=..., do_word_tokenize=..., do_subword_tokenize=..., word_tokenizer_type=..., subword_tokenizer_type=..., never_split=..., unk_token=..., sep_token=..., pad_token=..., cls_token=..., mask_token=..., mecab_kwargs=..., **kwargs) -> None:
        """Constructs a MecabBertTokenizer.

        Args:
            **vocab_file**: Path to a one-wordpiece-per-line vocabulary file.
            **do_lower_case**: (`optional`) boolean (default True)
                Whether to lower case the input.
                Only has an effect when do_basic_tokenize=True.
            **do_word_tokenize**: (`optional`) boolean (default True)
                Whether to do word tokenization.
            **do_subword_tokenize**: (`optional`) boolean (default True)
                Whether to do subword tokenization.
            **word_tokenizer_type**: (`optional`) string (default "basic")
                Type of word tokenizer.
            **subword_tokenizer_type**: (`optional`) string (default "wordpiece")
                Type of subword tokenizer.
            **mecab_kwargs**: (`optional`) dict passed to `MecabTokenizer` constructor (default None)
        """
        ...
    
    @property
    def do_lower_case(self):
        ...
    
    def __getstate__(self):
        ...
    
    def __setstate__(self, state):
        ...
    


class MecabTokenizer:
    """Runs basic tokenization with MeCab morphological parser."""
    def __init__(self, do_lower_case=..., never_split=..., normalize_text=..., mecab_dic: Optional[str] = ..., mecab_option: Optional[str] = ...) -> None:
        """Constructs a MecabTokenizer.

        Args:
            **do_lower_case**: (`optional`) boolean (default True)
                Whether to lowercase the input.
            **never_split**: (`optional`) list of str
                Kept for backward compatibility purposes.
                Now implemented directly at the base class level (see :func:`PreTrainedTokenizer.tokenize`)
                List of tokens not to split.
            **normalize_text**: (`optional`) boolean (default True)
                Whether to apply unicode normalization to text before tokenization.
            **mecab_dic**: (`optional`) string (default "ipadic")
                Name of dictionary to be used for MeCab initialization.
                If you are using a system-installed dictionary, set thi option to `None` and modify `mecab_option`.
            **mecab_option**: (`optional`) string
                String passed to MeCab constructor.
        """
        ...
    
    def tokenize(self, text, never_split=..., **kwargs):
        """Tokenizes a piece of text."""
        ...
    


class CharacterTokenizer:
    """Runs Character tokenziation."""
    def __init__(self, vocab, unk_token, normalize_text=...) -> None:
        """Constructs a CharacterTokenizer.

        Args:
            **vocab**:
                Vocabulary object.
            **unk_token**: str
                A special symbol for out-of-vocabulary token.
            **normalize_text**: (`optional`) boolean (default True)
                Whether to apply unicode normalization to text before tokenization.
        """
        ...
    
    def tokenize(self, text):
        """Tokenizes a piece of text into characters.

        For example:
            input = "apple"
            output = ["a", "p", "p", "l", "e"]
        Args:
            text: A single token or whitespace separated tokens.
                This should have already been passed through `BasicTokenizer`.
        Returns:
            A list of characters.
        """
        ...
    



"""
This type stub file was generated by pyright.
"""

import collections
import enum
import re
import pandas as pd
from dataclasses import dataclass
from typing import Callable, List, Optional, Text, Tuple, Union
from ...file_utils import add_end_docstrings, is_pandas_available
from ...tokenization_utils import PreTrainedTokenizer
from ...tokenization_utils_base import BatchEncoding, ENCODE_KWARGS_DOCSTRING, EncodedInput, ExplicitEnum, PaddingStrategy, PreTokenizedInput, TensorType, TextInput
from ...utils import logging

""" Tokenization class for TAPAS model."""
if is_pandas_available():
    ...
logger = logging.get_logger(__name__)
VOCAB_FILES_NAMES = { "vocab_file": "vocab.txt" }
PRETRAINED_VOCAB_FILES_MAP = { "vocab_file": { "google/tapas-large-finetuned-sqa": "https://huggingface.co/google/tapas-large-finetuned-sqa/resolve/main/vocab.txt","google/tapas-large-finetuned-wtq": "https://huggingface.co/google/tapas-large-finetuned-wtq/resolve/main/vocab.txt","google/tapas-large-finetuned-wikisql-supervised": "https://huggingface.co/google/tapas-large-finetuned-wikisql-supervised/resolve/main/vocab.txt","google/tapas-large-finetuned-tabfact": "https://huggingface.co/google/tapas-large-finetuned-tabfact/resolve/main/vocab.txt","google/tapas-base-finetuned-sqa": "https://huggingface.co/google/tapas-base-finetuned-sqa/resolve/main/vocab.txt","google/tapas-base-finetuned-wtq": "https://huggingface.co/google/tapas-base-finetuned-wtq/resolve/main/vocab.txt","google/tapas-base-finetuned-wikisql-supervised": "https://huggingface.co/google/tapas-base-finetuned-wikisql-supervised/resolve/main/vocab.txt","google/tapas-base-finetuned-tabfact": "https://huggingface.co/google/tapas-base-finetuned-tabfact/resolve/main/vocab.txt","google/tapas-medium-finetuned-sqa": "https://huggingface.co/google/tapas-medium-finetuned-sqa/resolve/main/vocab.txt","google/tapas-medium-finetuned-wtq": "https://huggingface.co/google/tapas-medium-finetuned-wtq/resolve/main/vocab.txt","google/tapas-medium-finetuned-wikisql-supervised": "https://huggingface.co/google/tapas-medium-finetuned-wikisql-supervised/resolve/main/vocab.txt","google/tapas-medium-finetuned-tabfact": "https://huggingface.co/google/tapas-medium-finetuned-tabfact/resolve/main/vocab.txt","google/tapas-small-finetuned-sqa": "https://huggingface.co/google/tapas-small-finetuned-sqa/resolve/main/vocab.txt","google/tapas-small-finetuned-wtq": "https://huggingface.co/google/tapas-small-finetuned-wtq/resolve/main/vocab.txt","google/tapas-small-finetuned-wikisql-supervised": "https://huggingface.co/google/tapas-small-finetuned-wikisql-supervised/resolve/main/vocab.txt","google/tapas-small-finetuned-tabfact": "https://huggingface.co/google/tapas-small-finetuned-tabfact/resolve/main/vocab.txt","google/tapas-tiny-finetuned-sqa": "https://huggingface.co/google/tapas-tiny-finetuned-sqa/resolve/main/vocab.txt","google/tapas-tiny-finetuned-wtq": "https://huggingface.co/google/tapas-tiny-finetuned-wtq/resolve/main/vocab.txt","google/tapas-tiny-finetuned-wikisql-supervised": "https://huggingface.co/google/tapas-tiny-finetuned-wikisql-supervised/resolve/main/vocab.txt","google/tapas-tiny-finetuned-tabfact": "https://huggingface.co/google/tapas-tiny-finetuned-tabfact/resolve/main/vocab.txt","google/tapas-mini-finetuned-sqa": "https://huggingface.co/google/tapas-mini-finetuned-sqa/resolve/main/vocab.txt","google/tapas-mini-finetuned-wtq": "https://huggingface.co/google/tapas-mini-finetuned-wtq/resolve/main/vocab.txt","google/tapas-mini-finetuned-wikisql-supervised": "https://huggingface.co/google/tapas-mini-finetuned-wikisql-supervised/resolve/main/vocab.txt","google/tapas-mini-finetuned-tabfact": "https://huggingface.co/google/tapas-mini-finetuned-tabfact/resolve/main/vocab.txt" } }
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = { name: 512 for name in PRETRAINED_VOCAB_FILES_MAP.keys() }
PRETRAINED_INIT_CONFIGURATION = { name: { "do_lower_case": True } for name in PRETRAINED_VOCAB_FILES_MAP.keys() }
class TapasTruncationStrategy(ExplicitEnum):
    """
    Possible values for the ``truncation`` argument in :meth:`~transformers.TapasTokenizer.__call__`. Useful for
    tab-completion in an IDE.
    """
    DROP_ROWS_TO_FIT = ...
    DO_NOT_TRUNCATE = ...


TableValue = collections.namedtuple("TokenValue", ["token", "column_id", "row_id"])
@dataclass(frozen=True)
class TokenCoordinates:
    column_index: int
    row_index: int
    token_index: int
    ...


@dataclass
class TokenizedTable:
    rows: List[List[List[Text]]]
    selected_tokens: List[TokenCoordinates]
    ...


@dataclass(frozen=True)
class SerializedExample:
    tokens: List[Text]
    column_ids: List[int]
    row_ids: List[int]
    segment_ids: List[int]
    ...


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    ...

def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    ...

TAPAS_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING = r"""
            add_special_tokens (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether or not to encode the sequences with the special tokens relative to their model.
            padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`False`):
                Activates and controls padding. Accepts the following values:

                * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a
                  single sequence if provided).
                * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
                  maximum acceptable input length for the model if that argument is not provided.
                * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
                  different lengths).
            truncation (:obj:`bool`, :obj:`str` or :class:`~transformers.TapasTruncationStrategy`, `optional`, defaults to :obj:`False`):
                Activates and controls truncation. Accepts the following values:

                * :obj:`True` or :obj:`'drop_rows_to_fit'`: Truncate to a maximum length specified with the argument
                  :obj:`max_length` or to the maximum acceptable input length for the model if that argument is not
                  provided. This will truncate row by row, removing rows from the table.
                * :obj:`False` or :obj:`'do_not_truncate'` (default): No truncation (i.e., can output batch with
                  sequence lengths greater than the model maximum admissible input size).
            max_length (:obj:`int`, `optional`):
                Controls the maximum length to use by one of the truncation/padding parameters.

                If left unset or set to :obj:`None`, this will use the predefined model maximum length if a maximum
                length is required by one of the truncation/padding parameters. If the model has no specific maximum
                input length (like XLNet) truncation/padding to a maximum length will be deactivated.
            is_split_into_words (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not the input is already pre-tokenized (e.g., split into words), in which case the tokenizer
                will skip the pre-tokenization step. This is useful for NER or token classification.
            pad_to_multiple_of (:obj:`int`, `optional`):
                If set will pad the sequence to a multiple of the provided value. This is especially useful to enable
                the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
            return_tensors (:obj:`str` or :class:`~transformers.tokenization_utils_base.TensorType`, `optional`):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                * :obj:`'tf'`: Return TensorFlow :obj:`tf.constant` objects.
                * :obj:`'pt'`: Return PyTorch :obj:`torch.Tensor` objects.
                * :obj:`'np'`: Return Numpy :obj:`np.ndarray` objects.
"""
class TapasTokenizer(PreTrainedTokenizer):
    r"""
    Construct a TAPAS tokenizer. Based on WordPiece. Flattens a table and one or more related sentences to be used by
    TAPAS models.

    This tokenizer inherits from :class:`~transformers.PreTrainedTokenizer` which contains most of the main methods.
    Users should refer to this superclass for more information regarding those methods.
    :class:`~transformers.TapasTokenizer` creates several token type ids to encode tabular structure. To be more
    precise, it adds 7 token type ids, in the following order: :obj:`segment_ids`, :obj:`column_ids`, :obj:`row_ids`,
    :obj:`prev_labels`, :obj:`column_ranks`, :obj:`inv_column_ranks` and :obj:`numeric_relations`:

    - segment_ids: indicate whether a token belongs to the question (0) or the table (1). 0 for special tokens and
      padding.
    - column_ids: indicate to which column of the table a token belongs (starting from 1). Is 0 for all question
      tokens, special tokens and padding.
    - row_ids: indicate to which row of the table a token belongs (starting from 1). Is 0 for all question tokens,
      special tokens and padding. Tokens of column headers are also 0.
    - prev_labels: indicate whether a token was (part of) an answer to the previous question (1) or not (0). Useful in
      a conversational setup (such as SQA).
    - column_ranks: indicate the rank of a table token relative to a column, if applicable. For example, if you have a
      column "number of movies" with values 87, 53 and 69, then the column ranks of these tokens are 3, 1 and 2
      respectively. 0 for all question tokens, special tokens and padding.
    - inv_column_ranks: indicate the inverse rank of a table token relative to a column, if applicable. For example, if
      you have a column "number of movies" with values 87, 53 and 69, then the inverse column ranks of these tokens are
      1, 3 and 2 respectively. 0 for all question tokens, special tokens and padding.
    - numeric_relations: indicate numeric relations between the question and the tokens of the table. 0 for all
      question tokens, special tokens and padding.

    :class:`~transformers.TapasTokenizer` runs end-to-end tokenization on a table and associated sentences: punctuation
    splitting and wordpiece.

    Args:
        vocab_file (:obj:`str`):
            File containing the vocabulary.
        do_lower_case (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to lowercase the input when tokenizing.
        do_basic_tokenize (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to do basic tokenization before WordPiece.
        never_split (:obj:`Iterable`, `optional`):
            Collection of tokens which will never be split during tokenization. Only has an effect when
            :obj:`do_basic_tokenize=True`
        unk_token (:obj:`str`, `optional`, defaults to :obj:`"[UNK]"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        sep_token (:obj:`str`, `optional`, defaults to :obj:`"[SEP]"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        pad_token (:obj:`str`, `optional`, defaults to :obj:`"[PAD]"`):
            The token used for padding, for example when batching sequences of different lengths.
        cls_token (:obj:`str`, `optional`, defaults to :obj:`"[CLS]"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        mask_token (:obj:`str`, `optional`, defaults to :obj:`"[MASK]"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        empty_token (:obj:`str`, `optional`, defaults to :obj:`"[EMPTY]"`):
            The token used for empty cell values in a table. Empty cell values include "", "n/a", "nan" and "?".
        tokenize_chinese_chars (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to tokenize Chinese characters. This should likely be deactivated for Japanese (see this
            `issue <https://github.com/huggingface/transformers/issues/328>`__).
        strip_accents: (:obj:`bool`, `optional`):
            Whether or not to strip all accents. If this option is not specified, then it will be determined by the
            value for :obj:`lowercase` (as in the original BERT).
        cell_trim_length (:obj:`int`, `optional`, defaults to -1):
            If > 0: Trim cells so that the length is <= this value. Also disables further cell trimming, should thus be
            used with :obj:`truncation` set to :obj:`True`.
        max_column_id (:obj:`int`, `optional`):
            Max column id to extract.
        max_row_id (:obj:`int`, `optional`):
            Max row id to extract.
        strip_column_names (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to add empty strings instead of column names.
        update_answer_coordinates (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to recompute the answer coordinates from the answer text.

    """
    vocab_files_names = ...
    pretrained_vocab_files_map = ...
    max_model_input_sizes = ...
    def __init__(self, vocab_file, do_lower_case=..., do_basic_tokenize=..., never_split=..., unk_token=..., sep_token=..., pad_token=..., cls_token=..., mask_token=..., empty_token=..., tokenize_chinese_chars=..., strip_accents=..., cell_trim_length: int = ..., max_column_id: int = ..., max_row_id: int = ..., strip_column_names: bool = ..., update_answer_coordinates: bool = ..., model_max_length: int = ..., additional_special_tokens: Optional[List[str]] = ..., **kwargs) -> None:
        ...
    
    @property
    def do_lower_case(self):
        ...
    
    @property
    def vocab_size(self):
        ...
    
    def get_vocab(self):
        ...
    
    def convert_tokens_to_string(self, tokens):
        """ Converts a sequence of tokens (string) in a single string. """
        ...
    
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = ...) -> Tuple[str]:
        ...
    
    def create_attention_mask_from_sequences(self, query_ids: List[int], table_values: List[TableValue]) -> List[int]:
        """
        Creates the attention mask according to the query token IDs and a list of table values.

        Args:
            query_ids (:obj:`List[int]`): list of token IDs corresponding to the ID.
            table_values (:obj:`List[TableValue]`): lift of table values, which are named tuples containing the
                token value, the column ID and the row ID of said token.

        Returns:
            :obj:`List[int]`: List of ints containing the attention mask values.
        """
        ...
    
    def create_segment_token_type_ids_from_sequences(self, query_ids: List[int], table_values: List[TableValue]) -> List[int]:
        """
        Creates the segment token type IDs according to the query token IDs and a list of table values.

        Args:
            query_ids (:obj:`List[int]`): list of token IDs corresponding to the ID.
            table_values (:obj:`List[TableValue]`): lift of table values, which are named tuples containing the
                token value, the column ID and the row ID of said token.

        Returns:
            :obj:`List[int]`: List of ints containing the segment token type IDs values.
        """
        ...
    
    def create_column_token_type_ids_from_sequences(self, query_ids: List[int], table_values: List[TableValue]) -> List[int]:
        """
        Creates the column token type IDs according to the query token IDs and a list of table values.

        Args:
            query_ids (:obj:`List[int]`): list of token IDs corresponding to the ID.
            table_values (:obj:`List[TableValue]`): lift of table values, which are named tuples containing the
                token value, the column ID and the row ID of said token.

        Returns:
            :obj:`List[int]`: List of ints containing the column token type IDs values.
        """
        ...
    
    def create_row_token_type_ids_from_sequences(self, query_ids: List[int], table_values: List[TableValue]) -> List[int]:
        """
        Creates the row token type IDs according to the query token IDs and a list of table values.

        Args:
            query_ids (:obj:`List[int]`): list of token IDs corresponding to the ID.
            table_values (:obj:`List[TableValue]`): lift of table values, which are named tuples containing the
                token value, the column ID and the row ID of said token.

        Returns:
            :obj:`List[int]`: List of ints containing the row token type IDs values.
        """
        ...
    
    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = ...) -> List[int]:
        """
        Build model inputs from a question and flattened table for question answering or sequence classification tasks
        by concatenating and adding special tokens.

        Args:
            token_ids_0 (:obj:`List[int]`): The ids of the question.
            token_ids_1 (:obj:`List[int]`, `optional`): The ids of the flattened table.

        Returns:
            :obj:`List[int]`: The model input with special tokens.
        """
        ...
    
    def get_special_tokens_mask(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = ..., already_has_special_tokens: bool = ...) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``prepare_for_model`` method.

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of question IDs.
            token_ids_1 (:obj:`List[int]`, `optional`):
                List of flattened table IDs.
            already_has_special_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            :obj:`List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        ...
    
    @add_end_docstrings(TAPAS_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    def __call__(self, table: pd.DataFrame, queries: Optional[Union[TextInput, PreTokenizedInput, EncodedInput, List[TextInput], List[PreTokenizedInput], List[EncodedInput]]] = ..., answer_coordinates: Optional[Union[List[Tuple], List[List[Tuple]]]] = ..., answer_text: Optional[Union[List[TextInput], List[List[TextInput]]]] = ..., add_special_tokens: bool = ..., padding: Union[bool, str, PaddingStrategy] = ..., truncation: Union[bool, str, TapasTruncationStrategy] = ..., max_length: Optional[int] = ..., pad_to_multiple_of: Optional[int] = ..., return_tensors: Optional[Union[str, TensorType]] = ..., return_token_type_ids: Optional[bool] = ..., return_attention_mask: Optional[bool] = ..., return_overflowing_tokens: bool = ..., return_special_tokens_mask: bool = ..., return_offsets_mapping: bool = ..., return_length: bool = ..., verbose: bool = ..., **kwargs) -> BatchEncoding:
        """
        Main method to tokenize and prepare for the model one or several sequence(s) related to a table.

        Args:
            table (:obj:`pd.DataFrame`):
                Table containing tabular data. Note that all cell values must be text. Use `.astype(str)` on a Pandas
                dataframe to convert it to string.
            queries (:obj:`str` or :obj:`List[str]`):
                Question or batch of questions related to a table to be encoded. Note that in case of a batch, all
                questions must refer to the **same** table.
            answer_coordinates (:obj:`List[Tuple]` or :obj:`List[List[Tuple]]`, `optional`):
                Answer coordinates of each table-question pair in the batch. In case only a single table-question pair
                is provided, then the answer_coordinates must be a single list of one or more tuples. Each tuple must
                be a (row_index, column_index) pair. The first data row (not the column header row) has index 0. The
                first column has index 0. In case a batch of table-question pairs is provided, then the
                answer_coordinates must be a list of lists of tuples (each list corresponding to a single
                table-question pair).
            answer_text (:obj:`List[str]` or :obj:`List[List[str]]`, `optional`):
                Answer text of each table-question pair in the batch. In case only a single table-question pair is
                provided, then the answer_text must be a single list of one or more strings. Each string must be the
                answer text of a corresponding answer coordinate. In case a batch of table-question pairs is provided,
                then the answer_coordinates must be a list of lists of strings (each list corresponding to a single
                table-question pair).
        """
        ...
    
    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, TAPAS_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    def batch_encode_plus(self, table: pd.DataFrame, queries: Optional[Union[List[TextInput], List[PreTokenizedInput], List[EncodedInput]]] = ..., answer_coordinates: Optional[List[List[Tuple]]] = ..., answer_text: Optional[List[List[TextInput]]] = ..., add_special_tokens: bool = ..., padding: Union[bool, str, PaddingStrategy] = ..., truncation: Union[bool, str, TapasTruncationStrategy] = ..., max_length: Optional[int] = ..., pad_to_multiple_of: Optional[int] = ..., return_tensors: Optional[Union[str, TensorType]] = ..., return_token_type_ids: Optional[bool] = ..., return_attention_mask: Optional[bool] = ..., return_overflowing_tokens: bool = ..., return_special_tokens_mask: bool = ..., return_offsets_mapping: bool = ..., return_length: bool = ..., verbose: bool = ..., **kwargs) -> BatchEncoding:
        """
        Prepare a table and a list of strings for the model.

        .. warning::
            This method is deprecated, ``__call__`` should be used instead.

        Args:
            table (:obj:`pd.DataFrame`):
                Table containing tabular data. Note that all cell values must be text. Use `.astype(str)` on a Pandas
                dataframe to convert it to string.
            queries (:obj:`List[str]`):
                Batch of questions related to a table to be encoded. Note that all questions must refer to the **same**
                table.
            answer_coordinates (:obj:`List[Tuple]` or :obj:`List[List[Tuple]]`, `optional`):
                Answer coordinates of each table-question pair in the batch. Each tuple must be a (row_index,
                column_index) pair. The first data row (not the column header row) has index 0. The first column has
                index 0. The answer_coordinates must be a list of lists of tuples (each list corresponding to a single
                table-question pair).
            answer_text (:obj:`List[str]` or :obj:`List[List[str]]`, `optional`):
                Answer text of each table-question pair in the batch. In case a batch of table-question pairs is
                provided, then the answer_coordinates must be a list of lists of strings (each list corresponding to a
                single table-question pair). Each string must be the answer text of a corresponding answer coordinate.
        """
        ...
    
    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING)
    def encode(self, table: pd.DataFrame, query: Optional[Union[TextInput, PreTokenizedInput, EncodedInput]] = ..., add_special_tokens: bool = ..., padding: Union[bool, str, PaddingStrategy] = ..., truncation: Union[bool, str, TapasTruncationStrategy] = ..., max_length: Optional[int] = ..., return_tensors: Optional[Union[str, TensorType]] = ..., **kwargs) -> List[int]:
        """
        Prepare a table and a string for the model. This method does not return token type IDs, attention masks, etc.
        which are necessary for the model to work correctly. Use that method if you want to build your processing on
        your own, otherwise refer to ``__call__``.

        Args:
            table (:obj:`pd.DataFrame`):
                Table containing tabular data. Note that all cell values must be text. Use `.astype(str)` on a Pandas
                dataframe to convert it to string.
            query (:obj:`str` or :obj:`List[str]`):
                Question related to a table to be encoded.
        """
        ...
    
    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, TAPAS_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    def encode_plus(self, table: pd.DataFrame, query: Optional[Union[TextInput, PreTokenizedInput, EncodedInput]] = ..., answer_coordinates: Optional[List[Tuple]] = ..., answer_text: Optional[List[TextInput]] = ..., add_special_tokens: bool = ..., padding: Union[bool, str, PaddingStrategy] = ..., truncation: Union[bool, str, TapasTruncationStrategy] = ..., max_length: Optional[int] = ..., pad_to_multiple_of: Optional[int] = ..., return_tensors: Optional[Union[str, TensorType]] = ..., return_token_type_ids: Optional[bool] = ..., return_attention_mask: Optional[bool] = ..., return_special_tokens_mask: bool = ..., return_offsets_mapping: bool = ..., return_length: bool = ..., verbose: bool = ..., **kwargs) -> BatchEncoding:
        """
        Prepare a table and a string for the model.

        Args:
            table (:obj:`pd.DataFrame`):
                Table containing tabular data. Note that all cell values must be text. Use `.astype(str)` on a Pandas
                dataframe to convert it to string.
            query (:obj:`str` or :obj:`List[str]`):
                Question related to a table to be encoded.
            answer_coordinates (:obj:`List[Tuple]` or :obj:`List[List[Tuple]]`, `optional`):
                Answer coordinates of each table-question pair in the batch. The answer_coordinates must be a single
                list of one or more tuples. Each tuple must be a (row_index, column_index) pair. The first data row
                (not the column header row) has index 0. The first column has index 0.
            answer_text (:obj:`List[str]` or :obj:`List[List[str]]`, `optional`):
                Answer text of each table-question pair in the batch. The answer_text must be a single list of one or
                more strings. Each string must be the answer text of a corresponding answer coordinate.
        """
        ...
    
    @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, TAPAS_ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    def prepare_for_model(self, raw_table: pd.DataFrame, raw_query: Union[TextInput, PreTokenizedInput, EncodedInput], tokenized_table: Optional[TokenizedTable] = ..., query_tokens: Optional[TokenizedTable] = ..., answer_coordinates: Optional[List[Tuple]] = ..., answer_text: Optional[List[TextInput]] = ..., add_special_tokens: bool = ..., padding: Union[bool, str, PaddingStrategy] = ..., truncation: Union[bool, str, TapasTruncationStrategy] = ..., max_length: Optional[int] = ..., pad_to_multiple_of: Optional[int] = ..., return_tensors: Optional[Union[str, TensorType]] = ..., return_token_type_ids: Optional[bool] = ..., return_attention_mask: Optional[bool] = ..., return_special_tokens_mask: bool = ..., return_offsets_mapping: bool = ..., return_length: bool = ..., verbose: bool = ..., prepend_batch_axis: bool = ..., **kwargs) -> BatchEncoding:
        """
        Prepares a sequence of input id so that it can be used by the model. It adds special tokens, truncates
        sequences if overflowing while taking into account the special tokens.

        Args:
            raw_table (:obj:`pd.DataFrame`):
                The original table before any transformation (like tokenization) was applied to it.
            raw_query (:obj:`TextInput` or :obj:`PreTokenizedInput` or :obj:`EncodedInput`):
                The original query before any transformation (like tokenization) was applied to it.
            tokenized_table (:obj:`TokenizedTable`):
                The table after tokenization.
            query_tokens (:obj:`List[str]`):
                The query after tokenization.
            answer_coordinates (:obj:`List[Tuple]` or :obj:`List[List[Tuple]]`, `optional`):
                Answer coordinates of each table-question pair in the batch. The answer_coordinates must be a single
                list of one or more tuples. Each tuple must be a (row_index, column_index) pair. The first data row
                (not the column header row) has index 0. The first column has index 0.
            answer_text (:obj:`List[str]` or :obj:`List[List[str]]`, `optional`):
                Answer text of each table-question pair in the batch. The answer_text must be a single list of one or
                more strings. Each string must be the answer text of a corresponding answer coordinate.
        """
        ...
    
    def get_answer_ids(self, column_ids, row_ids, tokenized_table, answer_texts_question, answer_coordinates_question):
        ...
    
    def convert_logits_to_predictions(self, data, logits, logits_agg=..., cell_classification_threshold=...):
        """
        Converts logits of :class:`~transformers.TapasForQuestionAnswering` to actual predicted answer coordinates and
        optional aggregation indices.

        The original implementation, on which this function is based, can be found `here
        <https://github.com/google-research/tapas/blob/4908213eb4df7aa988573350278b44c4dbe3f71b/tapas/experiments/prediction_utils.py#L288>`__.

        Args:
            data (:obj:`dict`):
                Dictionary mapping features to actual values. Should be created using
                :class:`~transformers.TapasTokenizer`.
            logits (:obj:`np.ndarray` of shape ``(batch_size, sequence_length)``):
                Tensor containing the logits at the token level.
            logits_agg (:obj:`np.ndarray` of shape ``(batch_size, num_aggregation_labels)``, `optional`):
                Tensor containing the aggregation logits.
            cell_classification_threshold (:obj:`float`, `optional`, defaults to 0.5):
                Threshold to be used for cell selection. All table cells for which their probability is larger than
                this threshold will be selected.

        Returns:
            :obj:`tuple` comprising various elements depending on the inputs:

            - predicted_answer_coordinates (``List[List[[tuple]]`` of length ``batch_size``): Predicted answer
              coordinates as a list of lists of tuples. Each element in the list contains the predicted answer
              coordinates of a single example in the batch, as a list of tuples. Each tuple is a cell, i.e. (row index,
              column index).
            - predicted_aggregation_indices (``List[int]``of length ``batch_size``, `optional`, returned when
              ``logits_aggregation`` is provided): Predicted aggregation operator indices of the aggregation head.
        """
        ...
    


class BasicTokenizer(object):
    """
    Constructs a BasicTokenizer that will run basic tokenization (punctuation splitting, lower casing, etc.).

    Args:
        do_lower_case (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to lowercase the input when tokenizing.
        never_split (:obj:`Iterable`, `optional`):
            Collection of tokens which will never be split during tokenization. Only has an effect when
            :obj:`do_basic_tokenize=True`
        tokenize_chinese_chars (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to tokenize Chinese characters.

            This should likely be deactivated for Japanese (see this `issue
            <https://github.com/huggingface/transformers/issues/328>`__).
        strip_accents: (:obj:`bool`, `optional`):
            Whether or not to strip all accents. If this option is not specified, then it will be determined by the
            value for :obj:`lowercase` (as in the original BERT).
    """
    def __init__(self, do_lower_case=..., never_split=..., tokenize_chinese_chars=..., strip_accents=...) -> None:
        ...
    
    def tokenize(self, text, never_split=...):
        """
        Basic Tokenization of a piece of text. Split on "white spaces" only, for sub-word tokenization, see
        WordPieceTokenizer.

        Args:
            **never_split**: (`optional`) list of str
                Kept for backward compatibility purposes. Now implemented directly at the base class level (see
                :func:`PreTrainedTokenizer.tokenize`) List of token not to split.
        """
        ...
    


class WordpieceTokenizer(object):
    """Runs WordPiece tokenization."""
    def __init__(self, vocab, unk_token, max_input_chars_per_word=...) -> None:
        ...
    
    def tokenize(self, text):
        """
        Tokenizes a piece of text into its word pieces. This uses a greedy longest-match-first algorithm to perform
        tokenization using the given vocabulary.

        For example, :obj:`input = "unaffable"` wil return as output :obj:`["un", "##aff", "##able"]`.

        Args:
          text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer`.

        Returns:
          A list of wordpiece tokens.
        """
        ...
    


class Relation(enum.Enum):
    HEADER_TO_CELL = ...
    CELL_TO_HEADER = ...
    QUERY_TO_HEADER = ...
    QUERY_TO_CELL = ...
    ROW_TO_CELL = ...
    CELL_TO_ROW = ...
    EQ = ...
    LT = ...
    GT = ...


@dataclass
class Date:
    year: Optional[int] = ...
    month: Optional[int] = ...
    day: Optional[int] = ...


@dataclass
class NumericValue:
    float_value: Optional[float] = ...
    date: Optional[Date] = ...


@dataclass
class NumericValueSpan:
    begin_index: int = ...
    end_index: int = ...
    values: List[NumericValue] = ...


@dataclass
class Cell:
    text: Text
    numeric_value: Optional[NumericValue] = ...


@dataclass
class Question:
    original_text: Text
    text: Text
    numeric_spans: Optional[List[NumericValueSpan]] = ...


_DateMask = collections.namedtuple("_DateMask", ["year", "month", "day"])
_YEAR = _DateMask(True, False, False)
_YEAR_MONTH = _DateMask(True, True, False)
_YEAR_MONTH_DAY = _DateMask(True, True, True)
_MONTH = _DateMask(False, True, False)
_MONTH_DAY = _DateMask(False, True, True)
_DATE_PATTERNS = (("%B", _MONTH), ("%Y", _YEAR), ("%Ys", _YEAR), ("%b %Y", _YEAR_MONTH), ("%B %Y", _YEAR_MONTH), ("%B %d", _MONTH_DAY), ("%b %d", _MONTH_DAY), ("%d %b", _MONTH_DAY), ("%d %B", _MONTH_DAY), ("%B %d, %Y", _YEAR_MONTH_DAY), ("%d %B %Y", _YEAR_MONTH_DAY), ("%m-%d-%Y", _YEAR_MONTH_DAY), ("%Y-%m-%d", _YEAR_MONTH_DAY), ("%Y-%m", _YEAR_MONTH), ("%B %Y", _YEAR_MONTH), ("%d %b %Y", _YEAR_MONTH_DAY), ("%Y-%m-%d", _YEAR_MONTH_DAY), ("%b %d, %Y", _YEAR_MONTH_DAY), ("%d.%m.%Y", _YEAR_MONTH_DAY), ("%A, %b %d", _MONTH_DAY), ("%A, %B %d", _MONTH_DAY))
_FIELD_TO_REGEX = (("%A", r"\w+"), ("%B", r"\w+"), ("%Y", r"\d{4}"), ("%b", r"\w{3}"), ("%d", r"\d{1,2}"), ("%m", r"\d{1,2}"))
_PROCESSED_DATE_PATTERNS = _process_date_patterns()
_MAX_DATE_NGRAM_SIZE = 5
_NUMBER_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"]
_ORDINAL_WORDS = ["zeroth", "first", "second", "third", "fourth", "fith", "sixth", "seventh", "eighth", "ninth", "tenth", "eleventh", "twelfth"]
_ORDINAL_SUFFIXES = ["st", "nd", "rd", "th"]
_NUMBER_PATTERN = re.compile(r"((^|\s)[+-])?((\.\d+)|(\d+(,\d\d\d)*(\.\d*)?))")
_MIN_YEAR = 1700
_MAX_YEAR = 2016
_INF = float("INF")
def get_all_spans(text, max_ngram_length):
    """
    Split a text into all possible ngrams up to 'max_ngram_length'. Split points are white space and punctuation.

    Args:
      text: Text to split.
      max_ngram_length: maximal ngram length.
    Yields:
      Spans, tuples of begin-end index.
    """
    ...

def normalize_for_match(text):
    ...

def format_text(text):
    """Lowercases and strips punctuation."""
    ...

def parse_text(text):
    """
    Extracts longest number and date spans.

    Args:
      text: text to annotate

    Returns:
      List of longest numeric value spans.
    """
    ...

_PrimitiveNumericValue = Union[float, Tuple[Optional[float], Optional[float], Optional[float]]]
_SortKeyFn = Callable[[NumericValue], Tuple[float, Ellipsis]]
_DATE_TUPLE_SIZE = 3
EMPTY_TEXT = "EMPTY"
NUMBER_TYPE = "number"
DATE_TYPE = "date"
def get_numeric_sort_key_fn(numeric_values):
    """
    Creates a function that can be used as a sort key or to compare the values. Maps to primitive types and finds the
    biggest common subset. Consider the values "05/05/2010" and "August 2007". With the corresponding primitive values
    (2010.,5.,5.) and (2007.,8., None). These values can be compared by year and date so we map to the sequence (2010.,
    5.), (2007., 8.). If we added a third value "2006" with primitive value (2006., None, None), we could only compare
    by the year so we would map to (2010.,), (2007.,) and (2006.,).

    Args:
     numeric_values: Values to compare

    Returns:
     A function that can be used as a sort key function (mapping numeric values to a comparable tuple)

    Raises:
      ValueError if values don't have a common type or are not comparable.
    """
    ...

def get_numeric_relation(value, other_value, sort_key_fn):
    """Compares two values and returns their relation or None."""
    ...

def add_numeric_values_to_question(question):
    """Adds numeric value spans to a question."""
    ...

def filter_invalid_unicode(text):
    """Return an empty string and True if 'text' is in invalid unicode."""
    ...

def filter_invalid_unicode_from_table(table):
    """
    Removes invalid unicode from table. Checks whether a table cell text contains an invalid unicode encoding. If yes,
    reset the table cell text to an empty str and log a warning for each invalid cell

    Args:
        table: table to clean.
    """
    ...

def add_numeric_table_values(table, min_consolidation_fraction=..., debug_info=...):
    """
    Parses text in table column-wise and adds the consolidated values. Consolidation refers to finding values with a
    common types (date or number)

    Args:
        table:
            Table to annotate.
        min_consolidation_fraction:
            Fraction of cells in a column that need to have consolidated value.
        debug_info:
            Additional information used for logging.
    """
    ...

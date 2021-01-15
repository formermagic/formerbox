from typing import List, Optional, Text, Tuple, Union

from formerbox.common.registrable import Registrable
from transformers import BatchEncoding
from typing_extensions import Literal, Protocol, runtime_checkable

TensorType = Literal["pt", "tf", "jax", "np"]

Truncation = Literal[
    "only_first",  # equals to `truncation=True`
    "only_second",
    "longest_first",
    "do_not_truncate",
]

Padding = Literal[
    "longest",  # equals to `padding=True`
    "max_length",
    "do_not_pad",
]

TextInput = Text
PreTokenizedInput = List[Text]
EncodedInput = List[int]

TextInputPair = Tuple[Text, Text]
PreTokenizedInputPair = Tuple[List[Text], List[Text]]
EncodedInputPair = Tuple[List[int], List[int]]


class TokenizerBase(Registrable):
    model_max_length: int


@runtime_checkable
class Tokenizer(Protocol):
    def __call__(
        self,
        text: Union[
            TextInput,
            PreTokenizedInput,
            List[TextInput],
            List[PreTokenizedInput],
        ],
        text_pair: Optional[
            Union[
                TextInput,
                PreTokenizedInput,
                List[TextInput],
                List[PreTokenizedInput],
            ]
        ] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, Text, Padding] = False,
        truncation: Union[bool, Text, Truncation] = False,
        max_length: Optional[int] = None,
        Textide: int = 0,
        is_split_into_words: bool = False,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[Text, TensorType]] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs,
    ) -> BatchEncoding:
        ...


@runtime_checkable
class Seq2SeqTokenizer(Tokenizer, Protocol):
    def prepare_seq2seq_batch(
        self,
        src_texts: List[Text],
        tgt_texts: Optional[List[Text]] = None,
        max_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
        padding: Union[Padding, bool] = "longest",
        truncation: Union[Truncation, bool] = True,
        return_tensors: Optional[TensorType] = None,
        **kwargs,
    ) -> BatchEncoding:
        ...

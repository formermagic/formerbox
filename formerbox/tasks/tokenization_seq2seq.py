from typing import List, Optional, Text, Union

from transformers import BatchEncoding
from typing_extensions import Literal, Protocol

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


class Seq2SeqTokenizer(Protocol):
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

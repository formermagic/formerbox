import os
import typing
from io import TextIOWrapper
from itertools import islice
from typing import Any, Iterable, Iterator, List, Optional, Text, Tuple

import numpy as np
import torch

T = typing.TypeVar("T")


def safe_round(number: Any, ndigits: int) -> float:
    round_number: float
    if hasattr(number, "__round__"):
        round_number = round(number, ndigits)
    elif torch.is_tensor(number) and number.numel() == 1:
        round_number = safe_round(number.item(), ndigits)
    elif np.ndim(number) == 0 and hasattr(number, "item"):
        round_number = safe_round(number.item(), ndigits)
    else:
        round_number = number
    return round_number


# pylint: disable=not-callable
def get_perplexity(
    loss: Optional[Any], ndigits: int = 2, base: int = 2
) -> torch.FloatTensor:
    ppl_tensor: torch.Tensor
    if loss is None:
        ppl_tensor = torch.tensor([0.0])
    try:
        ppl_tensor = torch.tensor([safe_round(base ** loss, ndigits)])
    except OverflowError:
        ppl_tensor = torch.tensor([float("inf")])

    return typing.cast(torch.FloatTensor, ppl_tensor.float())


def lines_in_file(filepath: Text) -> int:
    if not os.path.exists(filepath):
        raise FileNotFoundError(filepath)

    with open(filepath, mode="r", encoding="utf-8", errors="ignore") as input_file:
        blocks_iter = text_file_blocks_iter(input_file)
        n_lines = sum(block.count("\n") for block in blocks_iter)

    return n_lines


def text_file_blocks_iter(
    text_stream: TextIOWrapper, size: int = 65536
) -> Iterable[Text]:
    while True:
        block = text_stream.read(size)
        if not block:
            break
        yield block


def lookahead(iterable: Iterable) -> Iterable[Tuple[Any, bool]]:
    # get an iterator and pull the first value
    iterator = iter(iterable)
    last_item = next(iterator)  # pylint: disable=stop-iteration-return
    # run the iterator to exhaustion (starting from the second value)
    for val in iterator:
        # report the *previous* value (more to come)
        yield (last_item, True)
        last_item = val
    # report the last value
    yield (last_item, False)


def lazy_groups_of(iterable: Iterable[T], group_size: int) -> Iterator[List[T]]:
    """
    Takes an iterable and batches the individual instances into lists of the
    specified size. The last list may be smaller if there are instances left over.
    """
    iterator = iter(iterable)
    while True:
        _slice = list(islice(iterator, group_size))
        if len(_slice) > 0:
            yield _slice
        else:
            break

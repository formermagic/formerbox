import argparse
import inspect
import os
import typing
from io import TextIOWrapper
from pathlib import Path
from typing import Any, Dict, List, Optional, Text, Tuple, Type, Union

import numpy as np
import torch
from more_itertools import windowed

T = typing.TypeVar("T")  # pylint: disable=invalid-name


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
def perplexity(
    loss: Optional[Any], ndigits: int = 2, base: int = 2
) -> torch.FloatTensor:
    ppl_tensor: torch.FloatTensor
    if loss is None:
        return torch.FloatTensor([0.0])
    try:
        ppl_tensor = torch.FloatTensor([safe_round(base ** loss, ndigits)])
    except OverflowError:
        ppl_tensor = torch.FloatTensor([float("inf")])
    return ppl_tensor


def lines_in_file(filepath: Text) -> int:
    if not os.path.exists(filepath):
        raise FileNotFoundError(filepath)

    with open(filepath, mode="r", encoding="utf-8", errors="ignore") as input_file:
        blocks_iter = text_file_blocks_iter(input_file)
        n_lines = sum(block.count("\n") for block in blocks_iter)

    return n_lines


def text_file_blocks_iter(
    text_stream: TextIOWrapper, size: int = 65536
) -> typing.Iterable[Text]:
    while True:
        block = text_stream.read(size)
        if not block:
            break
        yield block


def lookahead(iterable: typing.Iterable) -> typing.Iterable[Tuple[Any, bool]]:
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


def lazy_groups_of(
    iterable: typing.Iterable[T], group_size: int
) -> typing.Iterable[List[T]]:
    """
    Takes an iterable and batches the individual instances into lists of the
    specified size. The last list may be smaller if there are instances left over.
    """
    for window in windowed(iterable, n=group_size, step=group_size):
        yield [x for x in window if x is not None]


def path_to_posix(path: Union[Text, Path]) -> Text:
    if isinstance(path, Path):
        return path.as_posix()
    return path


def all_subclasses(cls: Type[T]) -> typing.Iterable[Type[T]]:
    subclasses = set(cls.__subclasses__())
    for subclass in subclasses:
        _subclasses = all_subclasses(subclass)
        subclasses = subclasses.union(_subclasses)
    return subclasses  # type: ignore


def init_from_args(cls: Type[T]) -> Type[T]:
    def from_args(args: Dict[Text, Any], **kwargs: Any) -> T:
        valid_kwargs = inspect.signature(cls.__init__).parameters
        obj_kwargs = dict((name, args[name]) for name in valid_kwargs if name in args)
        obj_kwargs.update(**kwargs)
        return cls(**obj_kwargs)  # type: ignore

    setattr(cls, "from_args", from_args)
    return cls


def str2bool(string: Text) -> bool:
    if isinstance(string, bool):
        return string

    result: bool
    if string.lower() in ("yes", "true", "t", "y", "1"):
        result = True
    elif string.lower() in ("no", "false", "f", "n", "0"):
        result = False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

    return result


def iter_stide(
    iterable: typing.Iterable[T], chunk_size: int, stride: int
) -> typing.Iterable[List[T]]:
    assert chunk_size > stride, "stride must be less than chunk size"
    for window in windowed(iterable, n=chunk_size, step=chunk_size - stride):
        yield [x for x in window if x is not None]


def append_path_suffix(base_path: Union[Text, Path], suffix: Text) -> Text:
    if isinstance(base_path, Path):
        base_path = str(base_path)
    base_path, ext = os.path.splitext(base_path)
    return f"{base_path}{suffix}{ext}"


def update_left_inplace(left_dict: Dict[Any, Any], right_dict: Dict[Any, Any]) -> None:
    for key, item in right_dict.items():
        if key not in left_dict:
            left_dict[key] = item

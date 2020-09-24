import logging
import os
from abc import abstractmethod
from argparse import ArgumentParser
from io import TextIOWrapper
from typing import Any, Callable, List, Optional, Text, Tuple

import torch
from transformers import PreTrainedTokenizerFast
from typing_extensions import Literal

from gitnetic.common.registrable import Registrable
from gitnetic.data.indexed_dataset_setup import IndexedDatasetSetup
from gitnetic.utils import str2bool

logger = logging.getLogger(__name__)

Truncation = Literal[
    "only_first",
    "only_second",
    "longest_first",
    "do_not_truncate",
]


def dataset_dest_filepath(filepath_prefix: Text, extension: Text) -> Text:
    filename = os.path.basename(filepath_prefix)
    filename = filename.split(".")
    filename, exts = filename[0], filename[1:]

    dirname = os.path.dirname(filepath_prefix)
    exts = ".".join(exts)
    exts = f".{exts}" if exts else ""
    extension = f".{extension}" if extension else ""

    return os.path.join(dirname, f"{filename}{exts}{extension}")


def read_line(stream: TextIOWrapper) -> Text:
    position = stream.tell()
    while True:
        try:
            return stream.readline()
        except UnicodeDecodeError:
            position -= 1
            stream.seek(position)


def find_offsets(filename: Text, num_chunks: int) -> Tuple[int, List[int]]:
    with open(filename, mode="r", encoding="utf-8") as f:
        size = os.fstat(f.fileno()).st_size
        chunk_size = size // num_chunks
        offsets = [0 for _ in range(num_chunks + 1)]

        for i in range(1, num_chunks):
            f.seek(chunk_size * i)
            read_line(f)
            offsets[i] = f.tell()

    return size, offsets


class Binarizer(Registrable):
    # pylint: disable=too-many-arguments
    def __init__(
        self, dataset_setup: IndexedDatasetSetup, tokenizer: PreTrainedTokenizerFast
    ) -> None:
        self.dataset_setup = dataset_setup
        self.tokenizer = tokenizer

    def binarize_dataset(
        self,
        filename: Text,
        output_prefix: Text,
        start_offset: int,
        end_offset: int,
        **kwargs: Any,
    ) -> None:
        # prepare indexed dataset builder
        data_filepath = dataset_dest_filepath(output_prefix, extension="bin")
        index_filepath = dataset_dest_filepath(output_prefix, extension="idx")
        dataset_builder = self.dataset_setup.dataset_builder_type(
            data_filepath=data_filepath,
            index_filepath=index_filepath,
            dtype=self.dataset_setup.dataset_dtype,
            dataset_type=self.dataset_setup.dataset_type,
        )

        # convert text to ids and write to the data file
        self.binarize(
            filename=filename,
            tokenizer=self.tokenizer,
            consumer=dataset_builder.add_tokenized_ids,
            start_offset=start_offset,
            end_offset=end_offset,
            **kwargs,
        )

        # write meta data and type info
        dataset_builder.finalize()

    @abstractmethod
    def binarize(
        self,
        filename: Text,
        tokenizer: PreTrainedTokenizerFast,
        consumer: Callable[[torch.Tensor], None],
        start_offset: int = 0,
        end_offset: int = -1,
        **kwargs: Any,
    ) -> None:
        """Binarize the given chunk of file and pass to a consumer."""
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def add_argparse_args(parent_parser: ArgumentParser) -> ArgumentParser:
        raise NotImplementedError()


@Binarizer.register(name="flat-binarizer")
class FlatBinarizer(Binarizer):
    def binarize(
        self,
        filename: Text,
        tokenizer: PreTrainedTokenizerFast,
        consumer: Callable[[torch.Tensor], None],
        start_offset: int = 0,
        end_offset: int = -1,
        truncation: Truncation = "do_not_truncate",
        max_length: Optional[int] = None,
        stride: int = 0,
        return_overflowing_tokens: bool = False,
        **kwargs: Any,
    ) -> None:
        # pylint: disable=arguments-differ, too-many-locals, too-many-arguments
        del kwargs  # use only designated args

        # make sure we explicitly truncate if max_length is provided
        if max_length is not None and truncation == "do_not_truncate":
            truncation = "longest_first"

        with open(filename, mode="r", encoding="utf-8") as stream:
            stream.seek(start_offset)
            line = read_line(stream)
            while line:
                # remove all special chars
                line = line.rstrip()

                # check if end_offset is reached
                if end_offset and stream.tell() > end_offset:
                    break

                try:
                    # tokenize input text and feed to consumer handler
                    encoding = tokenizer(
                        line,
                        truncation=truncation,
                        max_length=max_length,
                        stride=stride,
                        return_overflowing_tokens=return_overflowing_tokens,
                    )

                    if encoding.input_ids:
                        if isinstance(encoding.input_ids[0], list):
                            for input_ids in encoding.input_ids:
                                consumer(torch.tensor(input_ids))
                        else:
                            consumer(torch.tensor(encoding.input_ids))

                except TypeError as err:
                    logger.warning("Unable to tokenize a text, error: %s", err)

                # get the next line to process
                line = stream.readline()

    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        truncation_choices = list(Truncation.__args__)  # type: ignore

        # fmt: off
        parser.add_argument("--truncation", type=str, choices=truncation_choices,
                            default="do_not_truncate", required=False,
                            help="")
        parser.add_argument("--max_length", type=int, default=None, required=False,
                            help="A maximum length of text sequence to encode.")
        parser.add_argument("--stride", type=int, default=0, required=False,
                            help="")
        parser.add_argument("--return_overflowing_tokens", type=str2bool, default=False,
                            required=False, help="")
        # fmt: on

        return parser

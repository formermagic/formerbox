import logging
import os
from abc import abstractmethod
from dataclasses import dataclass, field
from io import TextIOWrapper
from typing import Callable, List, Optional, Text, Tuple

import torch
from transformers import PreTrainedTokenizerFast
from typing_extensions import Literal

from gitnetic.common.dataclass_argparse import DataclassArgumentParser, DataclassBase
from gitnetic.common.registrable import ArgumentRegistrable
from gitnetic.data.indexed_dataset_setup import IndexedDatasetSetup

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


class Binarizer(ArgumentRegistrable):
    # pylint: disable=too-many-arguments
    @dataclass
    class Params(DataclassBase):
        ...

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
    ) -> None:
        """Binarize the given chunk of file and pass to a consumer."""
        raise NotImplementedError()


@Binarizer.register(name="flat-binarizer", constructor="from_args")
class FlatBinarizer(Binarizer):
    @dataclass
    class Params(DataclassBase):
        truncation: Truncation = field(
            default="do_not_truncate",
            metadata={"help": ""},
        )
        max_length: Optional[int] = field(
            default=None,
            metadata={"help": ""},
        )
        stride: int = field(
            default=0,
            metadata={"help": ""},
        )
        return_overflowing_tokens: bool = field(
            default=False,
            metadata={"help": ""},
        )

    def __init__(
        self,
        dataset_setup: IndexedDatasetSetup,
        tokenizer: PreTrainedTokenizerFast,
        params: Params,
    ) -> None:
        super().__init__(dataset_setup, tokenizer)
        self.params = params

    def binarize(
        self,
        filename: Text,
        tokenizer: PreTrainedTokenizerFast,
        consumer: Callable[[torch.Tensor], None],
        start_offset: int = 0,
        end_offset: int = -1,
    ) -> None:
        # pylint: disable=too-many-arguments
        # make sure we explicitly truncate if max_length is provided
        truncation = self.params.truncation
        if truncation == "do_not_truncate" and self.params.max_length is not None:
            truncation = "longest_first"

        # TODO: use a dataset reader instead of reading the file directly
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
                        truncation=self.params.truncation,
                        max_length=self.params.max_length,
                        stride=self.params.stride,
                        return_overflowing_tokens=self.params.return_overflowing_tokens,
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

    @classmethod
    def add_argparse_args(cls, parser: DataclassArgumentParser) -> None:
        parser.add_arguments(cls.Params)

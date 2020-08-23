import os
from io import TextIOWrapper
from typing import Callable, List, Optional, Text, Tuple

import torch
from transformers import PreTrainedTokenizerFast

from gitnetic.data.indexed_dataset_setup import IndexedDatasetSetup


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


class Binarizer:
    # pylint: disable=too-many-arguments
    def __init__(
        self,
        dataset_setup: IndexedDatasetSetup,
        tokenizer: PreTrainedTokenizerFast,
        tokenizer_max_length: Optional[int] = None,
    ) -> None:
        self.dataset_setup = dataset_setup
        self.tokenizer = tokenizer
        self.tokenizer_max_length = tokenizer_max_length

    def binarize_dataset(
        self, filename: Text, output_prefix: Text, start_offset: int, end_offset: int
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
            max_length=self.tokenizer_max_length,
        )

        # write meta data and type info
        dataset_builder.finalize()

    # pylint: disable=too-many-arguments
    @staticmethod
    def binarize(
        filename: Text,
        tokenizer: PreTrainedTokenizerFast,
        consumer: Callable[[torch.Tensor], None],
        start_offset: int = 0,
        end_offset: int = -1,
        max_length: Optional[int] = None,
    ) -> None:
        """Binarize the given chunk of file and pass to a consumer."""

        with open(filename, mode="r", encoding="utf-8") as f:
            f.seek(start_offset)
            line = read_line(f)
            while line:
                # check if end_offset is reached
                if end_offset and f.tell() > end_offset:
                    break

                # prepare tokenization params
                if max_length is not None:
                    kwargs = {"truncation": True, "max_length": max_length}
                else:
                    kwargs = {}

                # tokenize input text and feed to consumer handler
                line = line.rstrip()
                encoding = tokenizer(line, **kwargs)
                encoding = encoding.convert_to_tensors(tensor_type="pt")
                input_ids = encoding.input_ids.squeeze()
                consumer(input_ids)

                # get the next line to process
                line = f.readline()

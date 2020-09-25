import logging
import os
from abc import abstractmethod
from dataclasses import dataclass, field
from io import TextIOWrapper
from typing import Any, Callable, Dict, List, Optional, Text, Tuple

import torch
from datasets import load_dataset
from gitnetic.common.dataclass_argparse import DataclassArgumentParser, DataclassBase
from gitnetic.common.registrable import ArgumentRegistrable
from gitnetic.data.indexed_dataset_setup import IndexedDatasetSetup
from transformers import PreTrainedTokenizerFast
from typing_extensions import Literal

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
        self,
        dataset_setup: IndexedDatasetSetup,
        tokenizer: PreTrainedTokenizerFast,
    ) -> None:
        self.dataset_setup = dataset_setup
        self.tokenizer = tokenizer

    def binarize_dataset(self, filename: Text, output_prefix: Text) -> None:
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
        self.binarize(filename=filename, consumer=dataset_builder.add_tokenized_ids)

        logger.info("Finilizing the results")
        # write meta data and type info
        dataset_builder.finalize()

    @abstractmethod
    def binarize(
        self, filename: Text, consumer: Callable[[torch.Tensor], None]
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
        batched: bool = field(
            default=True,
            metadata={"help": ""},
        )
        batch_size: int = field(
            default=512,
            metadata={"help": ""},
        )
        num_proc: int = field(
            default=32,
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
        self, filename: Text, consumer: Callable[[torch.Tensor], None]
    ) -> None:
        dataset = load_dataset(
            path="text",
            data_files=[filename],
            split="train",
        )

        dataset = dataset.map(
            self.encode,
            batched=self.params.batched,
            batch_size=self.params.batch_size,
            num_proc=self.params.num_proc,
            remove_columns=["text"],
        )

        for instance in dataset:
            assert isinstance(instance, dict)
            input_ids = instance["input_ids"]

            if isinstance(input_ids[0], list):
                for ids in input_ids:
                    consumer(torch.tensor(ids))
            else:
                consumer(torch.tensor(input_ids))

    def encode(self, instance: Dict[Text, Any]) -> Dict[Text, Any]:
        result: List[List[int]] = []
        lines = instance["text"]
        assert isinstance(lines, list)

        # make sure we explicitly truncate if max_length is provided
        truncation = self.params.truncation
        if truncation == "do_not_truncate" and self.params.max_length is not None:
            truncation = "longest_first"

        try:
            # tokenize input text and feed to consumer handler
            encoding = self.tokenizer(
                lines,
                truncation=truncation,
                max_length=self.params.max_length,
                stride=self.params.stride,
                return_overflowing_tokens=self.params.return_overflowing_tokens,
            )

            result = encoding.input_ids

        except TypeError as err:
            logger.warning("Unable to tokenize a text, error: %s", err)

        return {"input_ids": result}

    @classmethod
    def add_argparse_args(cls, parser: DataclassArgumentParser) -> None:
        parser.add_arguments(cls.Params)

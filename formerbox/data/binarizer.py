import logging
import os
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from io import TextIOWrapper
from typing import Any, Dict, List, Optional, Text, Tuple, Type, Union

import torch
from datasets import Dataset, DatasetDict, load_dataset
from formerbox.common.dataclass_argparse import DataclassBase
from formerbox.common.has_params import HasParsableParams, ParamsType
from formerbox.common.registrable import Registrable
from formerbox.data.indexed_dataset import IndexedDatasetBuilderBase
from formerbox.data.indexed_dataset_setup import IndexedDatasetSetup
from transformers import PreTrainedTokenizerFast

logger = logging.getLogger(__name__)


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


class Binarizer(Registrable, HasParsableParams[ParamsType], metaclass=ABCMeta):
    params: ParamsType
    params_type: Type[ParamsType]

    def __init__(
        self,
        dataset_setup: IndexedDatasetSetup,
        tokenizer: PreTrainedTokenizerFast,
    ) -> None:
        super().__init__()
        self.dataset_setup = dataset_setup
        self.tokenizer = tokenizer

    @abstractmethod
    def binarize_dataset(
        self,
        filename: Text,
        output_prefix: Text,
        **kwargs: Any,
    ) -> None:
        raise NotImplementedError()

    @abstractmethod
    def binarize(self, filename: Text, **kwargs: Any) -> None:
        """Binarize the given chunk of file and pass to a consumer."""
        raise NotImplementedError()


class BinarizerBase(Binarizer, metaclass=ABCMeta):
    @dataclass
    class Params(DataclassBase):
        batched: bool = field(
            default=True,
            metadata={
                "help": "Whether or not to provide batches of examples to the function."
                " Default is set to `True`."
            },
        )
        batch_size: int = field(
            default=512,
            metadata={
                "help": "The number of examples per batch provided to function"
                " if batched=True batch_size <= 0 or batch_size == None:"
                " Provide the full dataset as a single batch to function."
                " Default is set to `512`."
            },
        )
        num_proc: int = field(
            default=8,
            metadata={
                "help": "The number of processes for multiprocessing."
                " Default is set to `8`."
            },
        )
        split_chunks: bool = field(
            default=False,
            metadata={
                "help": "Whether or not to write overlapping long document chunks"
                " as independent samples into the resulting dataset. You might"
                " want to use this strategy for preparing a language modeling dataset."
            },
        )

    params: Params
    params_type: Type[Params] = Params

    def __init__(
        self,
        dataset_setup: IndexedDatasetSetup,
        tokenizer: PreTrainedTokenizerFast,
        params: Params,
    ) -> None:
        super().__init__(dataset_setup, tokenizer)
        self.params = params

    @abstractmethod
    def encode(self, instance: Dict[Text, Any]) -> Dict[Text, Any]:
        raise NotImplementedError()

    def process_dataset(
        self,
        filename: Text,
        script_path: Text,
        script_version: Optional[Text],
        remove_columns: List[Text],
    ) -> Union[Dataset, DatasetDict]:
        # check if packaged scripts are set correctly
        if script_path in ["csv", "json", "text"]:
            if script_version is not None:
                logger.error(
                    "Script %s is packaged into datasets library."
                    " Make sure you do not set `script_version` argument.",
                    script_path,
                )

        dataset = load_dataset(
            path=script_path,
            data_files=[filename],
            split="train",
            script_version=script_version,
        )

        dataset = dataset.map(
            self.encode,
            batched=self.params.batched,
            batch_size=self.params.batch_size,
            num_proc=self.params.num_proc,
            remove_columns=remove_columns,
        )

        return dataset

    def write_instance(
        self, instance: Dict[Text, Any], consumer: IndexedDatasetBuilderBase
    ) -> None:
        # write each chunk as a sample if `input_ids` is a batch
        # and the `split_chunks` flag is set to true
        input_ids = instance["input_ids"]
        if isinstance(input_ids[0], list) and self.params.split_chunks:
            for ids in input_ids:
                consumer.add_tokenized_ids(torch.tensor(ids))
        else:
            consumer.add_tokenized_ids(torch.tensor(input_ids))

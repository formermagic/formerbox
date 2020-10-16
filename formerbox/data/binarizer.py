import logging
import os
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from io import TextIOWrapper
from typing import Any, Callable, Dict, List, Optional, Text, Tuple, Type, Union

import torch
from datasets import load_dataset
from formerbox.common.dataclass_argparse import DataclassBase
from formerbox.common.has_params import HasParsableParams, ParamsType
from formerbox.common.registrable import Registrable
from formerbox.data.indexed_dataset_setup import IndexedDatasetSetup
from formerbox.utils import iter_stide
from transformers import PreTrainedTokenizerFast
from typing_extensions import Literal

logger = logging.getLogger(__name__)

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
        logger.info("Processing the data file: %s", filename)
        self.binarize(filename=filename, consumer=dataset_builder.add_tokenized_ids)

        # write meta data and type info
        logger.info("Finilizing the results")
        dataset_builder.finalize()

    @abstractmethod
    def binarize(
        self, filename: Text, consumer: Callable[[torch.Tensor], None]
    ) -> None:
        """Binarize the given chunk of file and pass to a consumer."""
        raise NotImplementedError()


@Binarizer.register(name="transformer-binarizer", constructor="from_partial")
class TransformerBinarizer(Binarizer):
    @dataclass
    class Params(DataclassBase):
        truncation: Truncation = field(
            default="do_not_truncate",
            metadata={"help": "Activates and controls the truncation."},
        )
        padding: Padding = field(
            default="do_not_pad",
            metadata={"help": "Controls the padding strategy."},
        )
        max_length: Optional[int] = field(
            default=None,
            metadata={
                "help": "Pad to a maximum length specified with the argument"
                " max_length or to the maximum acceptable input length for"
                " the model if that argument is not provided."
            },
        )
        stride: int = field(
            default=0,
            metadata={
                "help": "If set to a number along with max_length, "
                " the overflowing tokens returned when return_overflowing_tokens=True"
                " will contain some tokens from the end of the truncated sequence"
                " returned to provide some overlap between truncated and overflowing sequences."
                " The value of this argument defines the number of overlapping tokens."
            },
        )
        return_overflowing_tokens: bool = field(
            default=False,
            metadata={"help": "Whether or not to return overflowing token sequences."},
        )
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

    def binarize(
        self, filename: Text, consumer: Callable[[torch.Tensor], None]
    ) -> None:
        dataset = load_dataset(
            path="text",
            data_files=[filename],
            split="train",
            script_version="master",
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

            # write each chunk as a sample if `input_ids` is a batch
            # and the `split_chunks` flag is set to true
            if isinstance(input_ids[0], list) and self.params.split_chunks:
                for ids in input_ids:
                    consumer(torch.tensor(ids))
            else:
                consumer(torch.tensor(input_ids))

    def encode(self, instance: Dict[Text, Any]) -> Dict[Text, Any]:
        result: List[Union[int, List[int]]] = []
        lines = instance["text"]

        # make sure we explicitly truncate if max_length is provided
        truncation = self.params.truncation
        if truncation == "do_not_truncate" and self.params.max_length is not None:
            truncation = "longest_first"

        try:
            # split long documents into smaller overlaping chunks
            # this is a workaround to fix tokenizers internal issues
            # with processing long sequences with special tokens
            batch = self.preprocess_batch(lines)
            # tokenize input text and feed to consumer handler
            encoding = self.tokenizer(
                batch,
                truncation=truncation,
                padding=self.params.padding,
                max_length=self.params.max_length,
                stride=self.params.stride,
                return_overflowing_tokens=self.params.return_overflowing_tokens,
            )

            result = encoding.input_ids

        except TypeError as err:
            logger.warning("Unable to tokenize a text, error: %s", err)

        return {"input_ids": result}

    def preprocess_batch(self, batch: Union[Text, List[Text]]) -> List[Text]:
        if not isinstance(batch, list):
            batch = [batch]

        result = []
        for instance in batch:
            sentences = iter_stide(instance.split(" "), chunk_size=512, stride=32)
            for sentence in sentences:
                result.append(" ".join(sentence))

        return result

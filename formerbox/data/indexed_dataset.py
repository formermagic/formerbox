import inspect
import logging
import os
import struct
from abc import ABCMeta, abstractmethod
from functools import lru_cache
from io import BufferedReader, BufferedWriter, FileIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Text, Type, Union

import numpy as np
import torch
from formerbox.utils import all_subclasses, path_to_posix
from numpy import float32, float64, int8, int16, int32, int64, uint8, uint16
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


element_codes: Dict[int, Any] = {
    1: uint8,
    2: int8,
    3: int16,
    4: int32,
    5: int64,
    6: float32,
    7: float64,
    8: uint16,
}


def element_code(dtype: np.dtype) -> int:
    for code, _dtype in element_codes.items():
        if _dtype == dtype:
            return code
    raise ValueError(dtype)


def read_longs(stream: Union[FileIO, BufferedReader], num: int) -> np.ndarray:
    buffer = np.empty(num, dtype=int64)
    stream.readinto(buffer)  # type: ignore
    return buffer


def write_longs(stream: Union[FileIO, BufferedWriter], buffer: List[int]) -> None:
    stream.write(np.array(buffer, dtype=int64))  # type: ignore


def make_index_filepath(prefix_path: Text) -> Text:
    return prefix_path + ".idx"


def make_data_filepath(prefix_path: Text) -> Text:
    return prefix_path + ".bin"


class IndexedDatasetBase(Dataset, metaclass=ABCMeta):
    """A base class for loading preprocessed binary datasets.
    Binary datasets are represented as 2 files (.bin, .idx),
    containing the data sequences (.bin) and indices (.idx).

    Attributes:
        dtype (:obj:`Optional[np.dtype]`): A numpy dtype for the written elements.
        length (:obj:`Optional[int]`): A total number of written sequences (aka lines).
        element_size (:obj:`Optional[int]`): A number of bytes for each written element.
        data_stream (:obj:`Optional[Union[FileIO, BufferedReader]]`):
            A FileIO stream for reading the data file (.bin).
        dim_offsets (:obj:`Optional[np.ndarray]`): A number of dimensions/axes
            for the given item (typically, tokenized text sequences have 1 dimension).
        data_offsets (:obj:`Optional[np.ndarray]`): A number of elements that
            preceed the given item's beginning.
        sizes (:obj:`Optional[np.ndarray]`): A number of elements for the given element
            written to the index (.idx) file.
    """

    magic_code: bytes = NotImplemented

    def __init__(self, filepath_prefix: Text, **kwargs: Any) -> None:
        # properties for reading data from files
        self.dtype: Optional[Type[np.dtype]] = None
        self.length: Optional[int] = None
        self.dim_offsets: Optional[np.ndarray] = None
        self.data_offsets: Optional[np.ndarray] = None
        self.sizes: Optional[np.ndarray] = None

        # properties for managing filenames
        self.filepath_prefix = filepath_prefix
        self.index_filepath = make_index_filepath(filepath_prefix)
        self.data_filepath = make_data_filepath(filepath_prefix)

        del kwargs  # the base class should accept any args

    # pylint: disable=invalid-length-returned
    def __len__(self) -> int:
        if self.length is None:
            raise ValueError("No length calculated at this moment.")
        return self.length

    @property
    @abstractmethod
    def supports_prefetch(self) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def prefetch(self, indices: List[int]) -> None:
        raise NotImplementedError()

    def validate_index(self, index: int) -> None:
        assert self.length is not None
        if index < 0 or index >= self.length:
            raise IndexError(f"Index({index}) is out of bounds")

    @staticmethod
    def from_file(
        filepath_prefix: Union[Text, Path], **kwargs: Any
    ) -> "IndexedDatasetBase":
        del kwargs  # support overrides

        filepath_prefix = path_to_posix(filepath_prefix)
        index_filepath = make_index_filepath(filepath_prefix)
        dataset: Optional[IndexedDatasetBase] = None

        candidates = all_subclasses(IndexedDatasetBase)
        with open(index_filepath, mode="rb") as stream:
            for candidate in candidates:
                assert issubclass(candidate, IndexedDatasetBase)
                if inspect.isabstract(candidate):
                    continue

                num_bytes = len(candidate.magic_code)
                magic_code = stream.read(num_bytes)
                if candidate.magic_code == magic_code:
                    dataset = candidate(filepath_prefix=filepath_prefix)  # type: ignore
                    break

                # return filedesc to the beginning
                stream.seek(0)

        if dataset is None:
            raise ValueError(
                "Unable to match a dataset with the given type."
                " Make sure you included `--dataset_impl` arg"
                " while preprocessing raw datasets."
            )

        return dataset


class IndexedDataset(IndexedDatasetBase):
    magic_code = b"ID\x00\x00"

    def __init__(self, filepath_prefix: Text) -> None:
        super().__init__(filepath_prefix)
        self.data_stream: Optional[Union[FileIO, BufferedReader]] = None
        self.read_index_file(self.index_filepath)

    def read_index_file(self, filepath: Text) -> None:
        with open(filepath, mode="rb") as index_file:
            magic_code = index_file.read(len(self.magic_code))
            assert (
                magic_code == self.magic_code
            ), "Index file doesn't match the expected format."
            code = struct.unpack("<B", index_file.read(1))[0]
            length, size = struct.unpack("<QQ", index_file.read(16))

            self.dtype = element_codes[code]
            self.length = length
            self.dim_offsets = read_longs(index_file, length + 1)
            self.data_offsets = read_longs(index_file, length + 1)
            self.sizes = read_longs(index_file, size)

    @property
    def supports_prefetch(self) -> bool:
        return False

    def prefetch(self, indices: List[int]) -> None:
        pass

    @lru_cache(maxsize=128)
    def __getitem__(self, index: int) -> torch.Tensor:
        # make sure the index is within bounds
        self.validate_index(index)
        # prepare the data file for reading
        if self.data_stream is None:
            self.data_stream = open(self.data_filepath, mode="rb", buffering=0)

        # a number of elements across all dimensions
        start, end = self.dim_offsets[index : index + 2]
        tensor_size = self.sizes[start:end]
        # a number of elements that preceed the current start element
        element_size = np.dtype(self.dtype).itemsize
        tensor_offset = self.data_offsets[index] * element_size

        buffer = np.fromfile(
            self.data_stream,
            dtype=self.dtype,
            count=tensor_size.prod(),
            offset=tensor_offset,
        )

        # get back the original offset
        self.data_stream.seek(0)

        buffer = buffer.reshape(tensor_size)
        item = torch.from_numpy(buffer).long()

        return item

    def __del__(self) -> None:
        if self.data_stream is not None:
            self.data_stream.close()


class IndexedCachedDataset(IndexedDataset):
    magic_code = b"ICD\x00\x00"

    def __init__(self, filepath_prefix: Text) -> None:
        super().__init__(filepath_prefix)
        self.cache: Optional[np.ndarray] = None
        self.cache_index: Dict[int, int] = {}

    @property
    def supports_prefetch(self) -> bool:
        return True

    def prefetch(self, indices: List[int]) -> None:
        # check if prefetching hasn't been done yet
        if all(idx in self.cache_index for idx in indices):
            return

        # prepare the data file for reading
        if self.data_stream is None:
            self.data_stream = open(self.data_filepath, mode="rb", buffering=0)

        indices = sorted(set(indices))

        # calculate the total number of elements to cache
        total_size = 0
        for idx in indices:
            total_size += self.data_offsets[idx + 1] - self.data_offsets[idx]

        self.cache = np.empty(total_size, dtype=self.dtype)
        self.cache_index.clear()

        size_offset = 0
        for idx in indices:
            self.cache_index[idx] = size_offset
            item_size = self.data_offsets[idx + 1] - self.data_offsets[idx]
            item_offset = self.data_offsets[idx] * self.dtype.itemsize  # type: ignore
            # read a buffer to cache from file
            buffer = np.fromfile(
                self.data_stream,
                dtype=self.dtype,
                count=item_size,
                offset=item_offset,
            )

            # get back the original offset
            self.data_stream.seek(0)

            # cache the read buffer
            self.cache[size_offset : size_offset + item_size] = buffer

            size_offset += item_size

        if self.data_stream is not None:
            self.data_stream.close()
            self.data_stream = None

    @lru_cache(maxsize=128)
    def __getitem__(self, index: int) -> torch.Tensor:
        # make sure the index is within bounds
        self.validate_index(index)

        # a number of elements across all dimensions
        start, end = self.dim_offsets[index : index + 2]
        tensor_size = self.sizes[start:end]

        # copy cached chunk into a buffer
        assert index in self.cache_index
        buffer = np.empty(tensor_size.prod(), dtype=self.dtype)
        size_offset = self.cache_index[index]
        np.copyto(buffer, self.cache[size_offset : size_offset + buffer.size])

        buffer = buffer.reshape(tensor_size)
        item = torch.from_numpy(buffer).long()

        return item


class IndexedDatasetBuilderBase:
    stream: Union[FileIO, BufferedWriter]

    def __init__(
        self,
        data_filepath: Text,
        index_filepath: Text,
        dtype: np.dtype,
        dataset_type: Type[IndexedDatasetBase],
    ) -> None:
        self.data_filepath = data_filepath
        self.index_filepath = index_filepath
        self.dtype = dtype
        self.dataset_type = dataset_type
        self.stream = open(self.data_filepath, mode="wb")

    @abstractmethod
    def add_tokenized_ids(self, input_ids: torch.Tensor) -> None:
        del input_ids
        raise NotImplementedError()

    @abstractmethod
    def finalize(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def merge_file(self, filepath_prefix: Text, remove_files: bool = True) -> None:
        del filepath_prefix, remove_files
        raise NotImplementedError()

    def __del__(self) -> None:
        # close the data stream if one is open
        if not self.stream.closed:
            self.stream.close()


class IndexedDatasetBuilder(IndexedDatasetBuilderBase):
    def __init__(
        self,
        data_filepath: Text,
        index_filepath: Text,
        dtype: np.dtype = np.dtype(int32),
        dataset_type: Type[IndexedDatasetBase] = IndexedDataset,
    ) -> None:
        super().__init__(data_filepath, index_filepath, dtype, dataset_type)
        self.data_offsets = [0]
        self.dim_offsets = [0]
        self.sizes = []

    def add_tokenized_ids(self, input_ids: torch.Tensor) -> None:
        # check if stream is open
        if self.stream.closed:
            self.stream = open(self.data_filepath, mode="wb")

        # write tokenized ids to data file
        input_ids_numpy = np.array(input_ids.numpy(), dtype=self.dtype)
        input_bytes = self.stream.write(input_ids_numpy.tobytes())
        input_size = input_bytes // self.dtype.itemsize

        # append updated offset for the added input
        self.data_offsets.append(self.data_offsets[-1] + input_size)
        # append tensor sizes by shape
        self.sizes.extend(input_ids.size())
        # append dim offsets for added tensor
        self.dim_offsets.append(self.dim_offsets[-1] + len(input_ids.size()))

    def finalize(self) -> None:
        # close the data stream if one is open
        if not self.stream.closed:
            self.stream.close()

        # write an index-specific metadata
        with open(self.index_filepath, mode="wb") as index_file:
            # write index format metadata
            index_file.write(self.dataset_type.magic_code)

            code = element_code(self.dtype)
            length = len(self.data_offsets) - 1
            size = len(self.sizes)

            # write sizes and types meta data
            index_file.write(struct.pack("<B", code))
            index_file.write(struct.pack("<QQ", length, size))

            # write long lists with offsets and sizes
            write_longs(index_file, self.dim_offsets)
            write_longs(index_file, self.data_offsets)
            write_longs(index_file, self.sizes)

    def merge_file(self, filepath_prefix: Text, remove_files: bool = True) -> None:
        # read indexed dataset to merge with the current one
        assert not inspect.isabstract(self.dataset_type)
        indexed_dataset = self.dataset_type(filepath_prefix=filepath_prefix)
        assert (
            indexed_dataset.dtype == self.dtype
        ), "Types must match for both datasets to be merged correctly."

        # merge `data_offsets` lists
        start_offset = self.data_offsets[-1]
        for offset in indexed_dataset.data_offsets[1:]:
            self.data_offsets.append(start_offset + offset)

        # merge `sizes` lists
        if indexed_dataset.sizes is not None:
            self.sizes.extend(indexed_dataset.sizes.tolist())

        # merge `dim_offsets` lists
        start_offset = self.dim_offsets[-1]
        for offset in indexed_dataset.dim_offsets[1:]:
            self.dim_offsets.append(start_offset + offset)

        # write data from indexed dataset file to the current data file
        with open(indexed_dataset.data_filepath, mode="rb") as data_file:
            while True:
                read_data = data_file.read(1024)
                if read_data:
                    self.stream.write(read_data)
                else:
                    break

        # remove merged temp files from disk if needed
        if remove_files:
            os.remove(indexed_dataset.data_filepath)
            os.remove(indexed_dataset.index_filepath)

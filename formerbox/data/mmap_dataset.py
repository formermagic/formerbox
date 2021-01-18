import logging
import os
import shutil
import struct
from abc import ABCMeta
from functools import lru_cache
from io import BufferedWriter, FileIO
from types import TracebackType
from typing import List, Optional, Text, Type, Union

import numpy as np
import torch
from formerbox.data.indexed_dataset import (
    IndexedDatasetBase,
    IndexedDatasetBuilderBase,
    element_code,
    element_codes,
)
from numpy import int32, int64
from torch import Tensor

logger = logging.getLogger(__name__)


class MMapIndexedDatasetBase(IndexedDatasetBase, metaclass=ABCMeta):
    def __init__(self, filepath_prefix: Text) -> None:
        super().__init__(filepath_prefix)
        self.index_buffer_mmap: Optional[np.memmap] = None
        self.index_buffer: Optional[memoryview] = None
        self.data_buffer_mmap: Optional[np.memmap] = None
        self.data_buffer: Optional[memoryview] = None


class MMapIndexedDataset(MMapIndexedDatasetBase):
    magic_code = b"MMID\x00\x00"

    def __init__(self, filepath_prefix: Text) -> None:
        super().__init__(filepath_prefix)
        self.read_index_file(self.index_filepath)
        self.read_data_file(self.data_filepath)

    def read_index_file(self, filepath: Text) -> None:
        with open(filepath, mode="rb") as index_file:
            magic_code = index_file.read(len(self.magic_code))
            assert (
                magic_code == self.magic_code
            ), "Index file doesn't match the expected format."
            code = struct.unpack("<B", index_file.read(1))[0]
            length, num_sizes = struct.unpack("<QQ", index_file.read(16))

            self.dtype = element_codes[code]
            self.length = length - 1  # has one extra (initial) item
            offset = index_file.tell()

        self.index_buffer_mmap = np.memmap(filepath, mode="r", order="C")  # type: ignore
        self.index_buffer = memoryview(self.index_buffer_mmap.data)
        self.dim_offsets = np.frombuffer(
            self.index_buffer,
            dtype=int32,
            count=self.length + 1,  # one for an initial item
            offset=offset,
        )

        offset += self.dim_offsets.nbytes
        self.sizes = np.frombuffer(
            self.index_buffer, dtype=int32, count=num_sizes, offset=offset
        )

        offset += self.sizes.nbytes
        self.data_offsets = np.frombuffer(
            self.index_buffer, dtype=int64, count=self.length, offset=offset
        )

    def read_data_file(self, filepath: Text) -> None:
        self.data_buffer_mmap = np.memmap(filepath, mode="r", order="C")  # type: ignore
        self.data_buffer = memoryview(self.data_buffer_mmap.data)

    @property
    def supports_prefetch(self) -> bool:
        return False

    def prefetch(self, indices: List[int]) -> None:
        pass

    @lru_cache(maxsize=128)
    def __getitem__(self, index: int) -> Tensor:
        # make sure the index is within bounds
        self.validate_index(index)

        # a number of elements across all dimensions
        start, end = self.dim_offsets[index : index + 2]
        tensor_size = self.sizes[start:end]
        # a pointer to the items' bytes in a file
        tensor_offset = self.data_offsets[index]

        buffer = np.frombuffer(
            self.data_buffer,
            dtype=self.dtype,
            count=tensor_size.prod(),
            offset=tensor_offset,
        )

        buffer = buffer.reshape(tensor_size)

        return torch.from_numpy(buffer.copy())

    def __del__(self) -> None:
        # close index mmap if needed
        if self.index_buffer_mmap is not None:
            self.index_buffer_mmap._mmap.close()
            del self.index_buffer_mmap

        # close data mmap if needed
        if self.data_buffer_mmap is not None:
            self.data_buffer_mmap._mmap.close()
            del self.data_buffer_mmap


class MMapIndexedDatasetBuilder(IndexedDatasetBuilderBase):
    def __init__(
        self,
        data_filepath: Text,
        index_filepath: Text,
        dtype: np.dtype = np.dtype(int64),
        dataset_type: Type[IndexedDatasetBase] = MMapIndexedDataset,
    ) -> None:
        super().__init__(data_filepath, index_filepath, dtype, dataset_type)
        self.dim_offsets = [0]
        self.sizes: List[int] = []

    def add_tokenized_ids(self, input_ids: Tensor) -> None:
        # check if stream is open
        if self.stream.closed:
            self.stream = open(self.data_filepath, mode="wb")

        # write the given input tensor as bytes
        buffer = np.array(input_ids.numpy(), dtype=self.dtype)
        self.stream.write(buffer.tobytes(order="C"))
        # extend sizes with all input dimensions
        self.sizes.extend(input_ids.size())
        # append the number of dimensions for the given input
        self.dim_offsets.append(self.dim_offsets[-1] + len(input_ids.size()))

    def finalize(self) -> None:
        # close the data stream if one is open
        if not self.stream.closed:
            self.stream.close()

        # write an index-specific metadata
        index_writer = IndexWriter(
            filepath=self.index_filepath,
            dtype=self.dtype,
            magic_code=self.dataset_type.magic_code,
        )

        with index_writer:
            index_writer.write(self.sizes, self.dim_offsets)

    def merge_file(self, filepath_prefix: Text, remove_files: bool = True) -> None:
        # read indexed dataset to merge with the current one
        indexed_dataset = MMapIndexedDataset(filepath_prefix)
        assert (
            indexed_dataset.dtype == self.dtype
        ), "Types must match for both datasets to be merged correctly."

        # merge `sizes` properties together
        if indexed_dataset.sizes is not None:
            self.sizes.extend(indexed_dataset.sizes.tolist())

        # merge `dim_offsets` properties together
        start_offset = self.dim_offsets[-1]
        for offset in indexed_dataset.dim_offsets[1:]:
            self.dim_offsets.append(start_offset + offset)

        # merge file contents into one file
        with open(indexed_dataset.data_filepath, mode="rb") as stream:
            assert self.stream is not None
            shutil.copyfileobj(stream, self.stream)  # type: ignore

        # remove merged temp files from disk if needed
        if remove_files:
            os.remove(indexed_dataset.data_filepath)
            os.remove(indexed_dataset.index_filepath)


class IndexWriter:
    def __init__(self, filepath: Text, dtype: np.dtype, magic_code: bytes) -> None:
        self.filepath = filepath
        self.dtype = dtype
        self.magic_code = magic_code
        self.stream: Optional[Union[FileIO, BufferedWriter]] = None

    def data_offsets(self, sizes: List[int], dim_offsets: List[int]) -> List[int]:
        offset = 0
        offsets: List[int] = []
        for idx in range(len(dim_offsets) - 1):
            # calculate the total number of elements
            start, end = dim_offsets[idx : idx + 2]
            size = np.prod(sizes[start:end])

            # append the pointer & pass the current element
            offsets.append(offset)
            offset += size * self.dtype.itemsize

        return offsets

    def write(self, sizes: List[int], dim_offsets: List[int]) -> None:
        # write `dim_offsets` and `sizes` lengths
        self.stream.write(struct.pack("<QQ", len(dim_offsets), len(sizes)))

        # prepare numpy buffers
        dim_offsets_buffer = np.array(dim_offsets, dtype=int32).tobytes(order="C")
        sizes_buffer = np.array(sizes, dtype=int32).tobytes(order="C")
        pointers = self.data_offsets(sizes, dim_offsets)
        data_offsets_buffer = np.array(pointers, dtype=int64).tobytes(order="C")

        # write numpy buffers
        self.stream.write(dim_offsets_buffer)
        self.stream.write(sizes_buffer)
        self.stream.write(data_offsets_buffer)

    def __enter__(self) -> "IndexWriter":
        self.stream = open(self.filepath, mode="wb")
        self.stream.write(self.magic_code)
        self.stream.write(struct.pack("<B", element_code(self.dtype)))
        return self

    def __exit__(
        self, exc_type: Type[Exception], exc_value: Exception, traceback: TracebackType
    ) -> None:
        del exc_type, exc_value, traceback
        self.stream.close()

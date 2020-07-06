import os
import struct
from functools import lru_cache
from io import TextIOWrapper
from typing import List, Optional, Text

import numpy as np
import torch
from torch.utils.data import Dataset

element_sizes = {
    np.uint8: 1,
    np.int8: 1,
    np.uint16: 2,
    np.int16: 2,
    np.int32: 4,
    np.int64: 8,
    np.float: 4,
    np.double: 8,
}

element_codes = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float,
    7: np.double,
    8: np.uint16,
}


def element_code(dtype: np.dtype) -> int:
    for code, _dtype in element_codes.items():
        if _dtype == dtype:
            return code
    raise ValueError(dtype)


def read_longs(stream: TextIOWrapper, num: int) -> np.array:
    buffer = np.empty(num, dtype=np.int64)
    stream.readinto(buffer)  # type: ignore
    return buffer


def write_longs(stream: TextIOWrapper, buffer: List[int]) -> None:
    stream.write(np.array(buffer, dtype=np.int64))


class IndexedDataset(Dataset):
    dtype: Optional[np.dtype] = None
    length: Optional[int] = None
    element_size: Optional[int] = None
    dim_offsets: Optional[np.array] = None
    data_offsets: Optional[np.array] = None
    sizes: Optional[np.array] = None
    data_stream: Optional[TextIOWrapper] = None

    def __init__(self, filepath_prefix: Text) -> None:
        super().__init__()
        self.filepath_prefix = filepath_prefix
        self.index_filepath = self._index_filepath(filepath_prefix)
        self.data_filepath = self._data_filepath(filepath_prefix)
        self._read_index_file(self.index_filepath)

    @staticmethod
    def _index_filepath(prefix_path: Text) -> Text:
        return prefix_path + ".idx"

    @staticmethod
    def _data_filepath(prefix_path: Text) -> Text:
        return prefix_path + ".bin"

    def _read_index_file(self, filepath: Text) -> None:
        with open(filepath, mode="rb") as f:
            code, element_size = struct.unpack("<QQ", f.read(16))
            length, size = struct.unpack("<QQ", f.read(16))

            self.dtype = element_codes[code]
            self.length = length
            self.element_size = element_size
            self.dim_offsets = read_longs(f, length + 1)  # type: ignore
            self.data_offsets = read_longs(f, length + 1)  # type: ignore
            self.sizes = read_longs(f, size)  # type: ignore

    def __len__(self) -> int:
        if self.length is None:
            return 0
        return self.length

    @lru_cache(maxsize=128)
    def __getitem__(self, index: int) -> torch.Tensor:
        if index < 0 or index >= self.length:
            raise IndexError("index out of range")
        if self.data_stream is None:
            self.data_stream = open(self.data_filepath, mode="rb", buffering=0)  # type: ignore

        start_idx = self.dim_offsets[index]
        end_idx = self.dim_offsets[index + 1]
        tensor_size = self.sizes[start_idx:end_idx]

        buffer = np.empty(tensor_size, dtype=self.dtype)

        offset = self.data_offsets[index] * self.element_size
        self.data_stream.seek(offset)
        self.data_stream.readinto(buffer)  # type: ignore

        item = torch.from_numpy(buffer).long()

        return item

    def __del__(self) -> None:
        if self.data_stream is not None:
            self.data_stream.close()


class IndexedDatasetBuilder:
    stream: Optional[TextIOWrapper] = None

    def __init__(self, data_filepath: Text, dtype: np.dtype = np.int32) -> None:
        self.data_filepath = data_filepath
        self.dtype = dtype
        self.data_offsets = [0]
        self.dim_offsets = [0]
        self.sizes = []

    @property
    def element_size(self) -> int:
        return element_sizes.get(self.dtype, 4)

    def add_tokenized_ids(self, input_ids: torch.Tensor) -> None:
        # check if stream is not none
        if self.stream is None:
            raise ValueError(
                "`stream` property must not be null, "
                "use `IndexedDatasetBuilder` instance for entering a context."
            )

        # write tokenized ids to data file
        input_ids_numpy = np.array(input_ids.numpy(), dtype=self.dtype)
        input_bytes = self.stream.write(input_ids_numpy)
        input_size = input_bytes // self.element_size

        # append updated offset for the added input
        self.data_offsets.append(self.data_offsets[-1] + input_size)
        # append tensor sizes by shape
        self.sizes.extend(input_ids.size())
        # append dim offsets for added tensor
        self.dim_offsets.append(self.dim_offsets[-1] + len(input_ids.size()))

    def merge_file(self, filepath_prefix: Text, remove_files: bool = True) -> None:
        # read indexed dataset to merge with the current one
        indexed_dataset = IndexedDataset(filepath_prefix)
        assert (
            indexed_dataset.dtype == self.dtype
        ), "Types must match for both datasets to be merged correctly."

        # merge `data_offsets` lists
        start_offset = self.data_offsets[-1]
        for offset in indexed_dataset.data_offsets[1:]:
            self.data_offsets.append(start_offset + offset)

        # merge `sizes` lists
        if indexed_dataset.sizes is not None:
            self.sizes.extend(indexed_dataset.sizes)

        # merge `dim_offsets` lists
        start_offset = self.dim_offsets[-1]
        for offset in indexed_dataset.dim_offsets[1:]:
            self.dim_offsets.append(start_offset + offset)

        # write data from indexed dataset file to the current data file
        with open(indexed_dataset.data_filepath, mode="rb") as data_file:
            while True:
                read_data = data_file.read(1024)
                if read_data:
                    self.stream.write(read_data)  # type: ignore
                else:
                    break

        # remove merged temp files from disk if needed
        if remove_files:
            os.remove(indexed_dataset.data_filepath)
            os.remove(indexed_dataset.index_filepath)

    def finalize(self, index_filepath: Text) -> None:
        with open(index_filepath, mode="wb") as index_file:
            code = element_code(self.dtype)
            length = len(self.data_offsets) - 1
            size = len(self.sizes)

            # write sizes and types meta data
            index_file.write(struct.pack("<QQ", code, self.element_size))
            index_file.write(struct.pack("<QQ", length, size))

            # write long lists with offsets and sizes
            write_longs(index_file, self.dim_offsets)  # type: ignore
            write_longs(index_file, self.data_offsets)  # type: ignore
            write_longs(index_file, self.sizes)  # type: ignore

    def __enter__(self) -> "IndexedDatasetBuilder":
        # file objects conform to `TextIOWrapper` type
        self.stream = open(self.data_filepath, mode="wb")  # type: ignore
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stream.close()

    def __del__(self) -> None:
        if self.stream is not None:
            self.stream.close()
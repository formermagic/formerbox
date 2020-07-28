import linecache
from typing import Any, Iterable, List, Text

import torch
from torch.utils.data import Dataset

from gitnetic.utils import lines_in_file

from .tokenization_codebert import CodeBertTokenizerFast


class DatasetWrapper(Dataset):
    def __iter__(self) -> Iterable[torch.Tensor]:
        raise NotImplementedError


class CodeBertDataset(Dataset):
    def __init__(
        self, tokenizer: CodeBertTokenizerFast, data_path: Text, max_length: int
    ) -> None:
        self.tokenizer = tokenizer
        self.data_path = data_path
        self.max_length = max_length
        self.num_sentences = lines_in_file(data_path)

    def __len__(self) -> int:
        return self.num_sentences

    def __getitem__(self, index: int) -> torch.Tensor:
        line = linecache.getline(self.data_path, index).rstrip()
        encoding = self.tokenizer(line, truncation=True, max_length=self.max_length)
        tensors = encoding.convert_to_tensors(tensor_type="pt")
        return tensors.input_ids.squeeze(0)


class CodeBertLazyDataset(DatasetWrapper):
    def __init__(
        self, tokenizer: CodeBertTokenizerFast, data_path: Text, max_length: int
    ) -> None:
        self.tokenizer = tokenizer
        self.data_path = data_path
        self.max_length = max_length
        self.num_sentences = lines_in_file(data_path)

    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError()

    def __iter__(self) -> Iterable[torch.Tensor]:
        with open(self.data_path, mode="r") as data_file:
            batch = []
            max_batch_len = 1024
            eof = False
            while not eof:
                try:
                    line = data_file.readline()
                    batch.append(line)
                except StopIteration:
                    eof = True

                if len(batch) >= max_batch_len or eof:
                    for tensor in self._batch_encode(batch):
                        yield tensor

    def _batch_encode(self, batch: List[Text]) -> torch.Tensor:
        batch_encoding = self.tokenizer(
            batch, truncation=True, padding=True  # , max_length=self.max_length
        )

        batch.clear()

        tensors = batch_encoding.convert_to_tensors(tensor_type="pt")
        return tensors.input_ids

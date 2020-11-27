from pathlib import Path
from typing import Any, Dict, List, Optional, Text, Union

from formerbox.data.indexed_dataset import IndexedDatasetBase
from torch import Tensor


# pylint: disable=arguments-differ
class Seq2SeqDataset(IndexedDatasetBase):
    magic_code = b"S2S\x00\x00"

    def __init__(
        self,
        filepath_prefix: Text,
        src_lang: Text,
        tgt_lang: Optional[Text] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(filepath_prefix, **kwargs)

        # load source and target lang datasets
        self.src_dataset = self.load_dataset(filepath_prefix, src_lang)
        self.tgt_dataset = None

        if tgt_lang is not None:
            # load target lang dataset if possible
            self.tgt_dataset = self.load_dataset(filepath_prefix, tgt_lang)
            # parallel datasets must be even
            self.validate_datasets()

        self.dtype = self.src_dataset.dtype
        self.length = self.src_dataset.length
        self.dim_offsets = self.src_dataset.dim_offsets
        self.data_offsets = self.src_dataset.data_offsets
        self.sizes = self.src_dataset.sizes

        self.kwargs = kwargs

    def __getitem__(self, index: int) -> Dict[Text, Tensor]:
        src_input_ids = self.src_dataset[index]
        if self.tgt_dataset is None:
            return {"src_input_ids": src_input_ids}

        tgt_input_ids = self.tgt_dataset[index]
        return {"src_input_ids": src_input_ids, "tgt_input_ids": tgt_input_ids}

    @property
    def supports_prefetch(self) -> bool:
        if self.tgt_dataset is None:
            return self.src_dataset.supports_prefetch

        return self.src_dataset.supports_prefetch and self.tgt_dataset.supports_prefetch

    def prefetch(self, indices: List[int]) -> None:
        self.src_dataset.prefetch(indices)
        self.tgt_dataset.prefetch(indices)

    @staticmethod
    def from_file(
        filepath_prefix: Union[Text, Path],
        src_lang: Text,
        tgt_lang: Optional[Text] = None,
        **kwargs: Any,
    ) -> "Seq2SeqDataset":
        del kwargs  # use designated args
        if isinstance(filepath_prefix, Path):
            filepath_prefix = str(filepath_prefix)
        return Seq2SeqDataset(
            filepath_prefix=filepath_prefix,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
        )

    def load_dataset(self, filepath_prefix: Text, lang: Text) -> IndexedDatasetBase:
        """Load a certain dataset using its file prefix and language extension."""
        filename = f"{filepath_prefix}.{lang}"
        return IndexedDatasetBase.from_file(filename)

    def validate_datasets(self) -> None:
        """Ensure if source and target datasets can be composed together."""
        assert self.src_dataset.dtype == self.tgt_dataset.dtype
        assert self.src_dataset.length == self.tgt_dataset.length

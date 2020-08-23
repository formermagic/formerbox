from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from typing import Any, Dict, Text, Type, Union

import numpy as np

from gitnetic.data.indexed_dataset import (
    IndexedCachedDataset,
    IndexedDataset,
    IndexedDatasetBuilder,
    IndexedDatasetBuilderMixin,
    IndexedDatasetMixin,
)
from gitnetic.data.mmap_dataset import MMapIndexedDataset, MMapIndexedDatasetBuilder


@dataclass
class IndexedDatasetSetup:
    dataset_builder_type: Type[IndexedDatasetBuilderMixin]
    dataset_type: Type[IndexedDatasetMixin]
    dataset_dtype: np.dtype

    @staticmethod
    def add_arguments(parser: ArgumentParser) -> None:
        parser.add_argument(
            "--dataset_impl",
            type=str,
            default=None,
            required=True,
            choices=["lazy", "cached", "mmap"],
            help="",
        )

    # pylint: disable=no-else-return
    @staticmethod
    def from_args(
        args: Union[Namespace, Dict[Text, Any], Text]
    ) -> "IndexedDatasetSetup":
        if isinstance(args, Namespace):
            args = vars(args)
        elif isinstance(args, str):
            args = {"dataset_impl": args}

        assert "dataset_impl" in args, "Unable to find the `dataset_impl` argument."

        dataset_impl = args["dataset_impl"]
        if dataset_impl == "lazy":
            return IndexedDatasetSetup(
                dataset_builder_type=IndexedDatasetBuilder,
                dataset_type=IndexedDataset,
                dataset_dtype=np.dtype(np.int32),
            )
        elif dataset_impl == "cached":
            return IndexedDatasetSetup(
                dataset_builder_type=IndexedDatasetBuilder,
                dataset_type=IndexedCachedDataset,
                dataset_dtype=np.dtype(np.int32),
            )
        elif dataset_impl == "mmap":
            return IndexedDatasetSetup(
                dataset_builder_type=MMapIndexedDatasetBuilder,
                dataset_type=MMapIndexedDataset,
                dataset_dtype=np.dtype(np.int64),
            )

        raise ValueError("Unable to match the given dataset type.")

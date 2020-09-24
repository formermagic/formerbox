from argparse import Namespace
from dataclasses import dataclass, field
from typing import Any, Dict, Text, Type, Union

import numpy as np
from typing_extensions import Literal

from gitnetic.common.dataclass_argparse import DataclassArgumentParser, DataclassBase
from gitnetic.common.registrable import ArgumentRegistrable
from gitnetic.data.indexed_dataset import (
    IndexedCachedDataset,
    IndexedDataset,
    IndexedDatasetBase,
    IndexedDatasetBuilder,
    IndexedDatasetBuilderBase,
)
from gitnetic.data.mmap_dataset import MMapIndexedDataset, MMapIndexedDatasetBuilder


class IndexedDatasetSetup(ArgumentRegistrable):
    # pylint: disable=arguments-differ
    @dataclass
    class Params(DataclassBase):
        dataset_impl: Literal["lazy", "cached", "mmap"] = field(
            metadata={"help": "Determines the type of a dataset to build."},
        )

    def __init__(
        self,
        dataset_builder_type: Type[IndexedDatasetBuilderBase],
        dataset_type: Type[IndexedDatasetBase],
        dataset_dtype: np.dtype,
    ) -> None:
        self.dataset_builder_type = dataset_builder_type
        self.dataset_type = dataset_type
        self.dataset_dtype = dataset_dtype

    @classmethod
    def from_args(
        cls, args: Union[Namespace, Dict[Text, Any], Text, Params], **kwargs: Any
    ) -> "IndexedDatasetSetup":
        del kwargs  # use only designated args
        if isinstance(args, Namespace):
            args = vars(args)
        elif isinstance(args, str):
            args = {"dataset_impl": args}
        elif isinstance(args, cls.Params):
            args = {"dataset_impl": args.dataset_impl}

        assert "dataset_impl" in args, "Unable to find the `dataset_impl` argument."

        result: IndexedDatasetSetup
        dataset_impl = args["dataset_impl"]
        if dataset_impl == "lazy":
            result = IndexedDatasetSetup(
                dataset_builder_type=IndexedDatasetBuilder,
                dataset_type=IndexedDataset,
                dataset_dtype=np.dtype(np.int32),
            )
        elif dataset_impl == "cached":
            result = IndexedDatasetSetup(
                dataset_builder_type=IndexedDatasetBuilder,
                dataset_type=IndexedCachedDataset,
                dataset_dtype=np.dtype(np.int32),
            )
        elif dataset_impl == "mmap":
            result = IndexedDatasetSetup(
                dataset_builder_type=MMapIndexedDatasetBuilder,
                dataset_type=MMapIndexedDataset,
                dataset_dtype=np.dtype(np.int64),
            )
        else:
            raise ValueError("Unable to match the given dataset type.")

        return result

    @classmethod
    def add_argparse_args(cls, parser: DataclassArgumentParser) -> None:
        parser.add_arguments(cls.Params)

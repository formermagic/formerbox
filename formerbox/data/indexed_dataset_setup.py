import logging
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Text, Type, Union

import numpy as np
from formerbox.common.dataclass_argparse import MISSING, DataclassBase
from formerbox.common.has_params import HasParsableParams
from formerbox.data.indexed_dataset import (
    IndexedCachedDataset,
    IndexedDataset,
    IndexedDatasetBase,
    IndexedDatasetBuilder,
    IndexedDatasetBuilderBase,
)
from formerbox.data.mmap_dataset import MMapIndexedDataset, MMapIndexedDatasetBuilder
from numpy import int32, int64
from typing_extensions import Literal

logger = logging.getLogger(__name__)


class IndexedDatasetSetup(HasParsableParams):
    @dataclass
    class Params(DataclassBase):
        dataset_impl: Literal["lazy", "cached", "mmap"] = field(
            default=MISSING,
            metadata={"help": "Determines the type of a dataset to build."},
        )

    params: Optional[Params]
    params_type = Params

    def __init__(
        self,
        dataset_builder_type: Type[IndexedDatasetBuilderBase],
        dataset_type: Type[IndexedDatasetBase],
        dataset_dtype: np.dtype,
    ) -> None:
        super().__init__()
        self.dataset_builder_type = dataset_builder_type
        self.dataset_type = dataset_type
        self.dataset_dtype = dataset_dtype
        self.params = None

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
                dataset_dtype=np.dtype(int32),
            )
        elif dataset_impl == "cached":
            result = IndexedDatasetSetup(
                dataset_builder_type=IndexedDatasetBuilder,
                dataset_type=IndexedCachedDataset,
                dataset_dtype=np.dtype(int32),
            )
        elif dataset_impl == "mmap":
            result = IndexedDatasetSetup(
                dataset_builder_type=MMapIndexedDatasetBuilder,
                dataset_type=MMapIndexedDataset,
                dataset_dtype=np.dtype(int64),
            )
        else:
            raise ValueError("Unable to match the given dataset type.")

        return result

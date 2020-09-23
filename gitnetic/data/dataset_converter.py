from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Text, Union

from datasets import Dataset, DatasetDict, load_dataset

from gitnetic.common.dataclass_argparse import DataclassArgumentParser, DataclassBase
from gitnetic.common.registrable import ArgumentRegistrable
from gitnetic.utils import lazy_groups_of
from gitnetic.utils.code_tokenizer import tokenize_python


class DatasetProcessingMixin:
    def filter_fn(self, instance: Dict[Text, Any], column: Text) -> bool:
        return instance[column] is not None

    def distinct_indices(self, dataset: Dataset, column: Text) -> List[int]:
        df = dataset.data.column(column).to_pandas()
        return df.drop_duplicates().index.to_list()

    def distinct_dataset(
        self,
        dataset: Union[Dataset, DatasetDict],
        column: Text,
        num_proc: int = 1,
    ) -> Union[Dataset, DatasetDict]:
        dataset = dataset.filter(
            self.filter_fn,
            num_proc=num_proc,
            fn_kwargs={"column": column},
        )

        result: Union[Dataset, DatasetDict]
        if isinstance(dataset, DatasetDict):
            indices = {
                k: self.distinct_indices(dataset, column)
                for k, dataset in dataset.items()
            }
            result = DatasetDict(
                {k: dataset.select(indices[k]) for k, dataset in dataset.items()}
            )
        else:
            indices = self.distinct_indices(dataset, column)
            result = dataset.select(indices)

        return result


class DatasetConverter(ArgumentRegistrable):
    @abstractmethod
    def convert(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError()

    @abstractmethod
    def encode(self, instance: Dict[Text, Any]) -> Dict[Text, Any]:
        raise NotImplementedError()

    @abstractmethod
    def save_dataset(
        self, dataset: Union[Dataset, DatasetDict], output_path: Text
    ) -> None:
        raise NotImplementedError()

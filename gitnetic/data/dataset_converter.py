import logging
import os
from abc import abstractmethod
from dataclasses import dataclass, field
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Text, Union

from datasets import Dataset, DatasetDict, load_dataset
from gitnetic.common.dataclass_argparse import DataclassArgumentParser, DataclassBase
from gitnetic.common.registrable import ArgumentRegistrable
from gitnetic.utils import lazy_groups_of
from gitnetic.utils.code_tokenizer import tokenize_python
from typeguard import typechecked

logger = logging.getLogger(__name__)

Instance = Optional[Text]


class DatasetProcessingMixin:
    def filter_fn(self, instance: Dict[Text, Any], column: Text) -> bool:
        return instance[column] is not None

    def distinct_indices(self, dataset: Dataset, column: Text) -> List[int]:
        df = dataset.data.column(column).to_pandas()
        return df.drop_duplicates().index.to_list()

    @typechecked
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
    @dataclass
    class Params(DataclassBase):
        ...

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


@DatasetConverter.register("code-lm-converter", constructor="from_args")
class CodeLMDatasetConverter(DatasetConverter, DatasetProcessingMixin):
    @dataclass
    class Params(DataclassBase):
        script_path: Text = field(metadata={"help": ""})
        output_path: Text = field(metadata={"help": ""})
        data_files: List[Text] = field(metadata={"help": ""})
        train_test_split: bool = field(default=False, metadata={"help": ""})
        train_size: float = field(default=0.8, metadata={"help": ""})
        valid_size: float = field(default=0.1, metadata={"help": ""})
        test_size: float = field(default=0.1, metadata={"help": ""})
        batched: bool = field(default=True, metadata={"help": ""})
        batch_size: int = field(default=128, metadata={"help": ""})
        num_proc: int = field(default=1, metadata={"help": ""})

    @typechecked
    def __init__(self, params: Params) -> None:
        self.params = params

    @typechecked
    def convert(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs  # use only designated args
        # search data_files if a user specified a pattern to use
        # otherwise the method will return the input values
        data_files = self.search_data_files(self.params.data_files)
        # prepare the output dir for writing results
        Path(self.params.output_path).parent.mkdir(exist_ok=True)

        dataset = load_dataset(
            self.params.script_path,
            data_files=data_files,
            split="train",
        )
        dataset = self.distinct_dataset(
            dataset,
            column="content",
            num_proc=self.params.num_proc,
        )
        dataset = dataset.map(
            self.encode,
            batched=self.params.batched,
            batch_size=self.params.batch_size,
            num_proc=self.params.num_proc,
        )

        def append_path_suffix(base_path: Text, suffix: Text) -> Text:
            base_path, ext = os.path.splitext(base_path)
            return f"{base_path}{suffix}{ext}"

        train_dataset = dataset
        if self.params.train_test_split:
            # sample and save the validation dataset
            if self.params.valid_size > 0:
                dataset_split = train_dataset.train_test_split(
                    test_size=self.params.valid_size
                )

                train_dataset, valid_dataset = dataset_split.values()
                valid_path = append_path_suffix(self.params.output_path, ".valid")
                self.save_dataset(valid_dataset, valid_path)

            # sample and save the test dataset
            if self.params.test_size > 0:
                dataset_split = train_dataset.train_test_split(
                    test_size=self.params.test_size
                )

                train_dataset, test_dataset = dataset_split.values()
                test_path = append_path_suffix(self.params.output_path, ".test")
                self.save_dataset(test_dataset, test_path)

            # sample all remaining instances to the training dataset
            train_path = append_path_suffix(self.params.output_path, ".train")
            self.save_dataset(train_dataset, train_path)
        else:
            self.save_dataset(dataset, self.params.output_path)

    def encode(self, instance: Dict[Text, Any]) -> Dict[Text, Any]:
        result: Union[Instance, List[Instance]]
        content = instance["content"]
        if isinstance(content, list):
            result = [self.tokenize_text(text) for text in content]
        else:
            result = self.tokenize_text(content)
        return {"output_data": result}

    @typechecked
    def save_dataset(
        self, dataset: Union[Dataset, DatasetDict], output_path: Text
    ) -> None:
        data: List[Text] = dataset["output_data"]  # type: ignore
        with open(output_path, mode="w") as stream:
            for group in lazy_groups_of(data, group_size=1000):
                instances = [instance for instance in group if instance]
                stream.write("\n".join(instances))

    @classmethod
    @typechecked
    def add_argparse_args(cls, parser: DataclassArgumentParser) -> None:
        parser.add_arguments(cls.Params)

    @typechecked
    def tokenize_text(self, text: Text) -> Instance:
        tokens = tokenize_python(text, keep_comments=True)
        result = " ".join(tokens)
        if not result:
            return None
        return result

    def search_data_files(self, data_files: List[Text]) -> List[Text]:
        assert data_files
        if len(data_files) > 1:
            return data_files

        search_pattern = data_files[0]
        search_results = glob(search_pattern)
        if search_results:
            logger.info(
                "Found %d files using the pattern: %s",
                len(search_results),
                search_pattern,
            )

            data_files = search_results

        return data_files

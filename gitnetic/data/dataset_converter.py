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


@DatasetConverter.register("code-lm-converter", constructor="from_args")
class CodeLMDatasetConverter(DatasetConverter, DatasetProcessingMixin):
    @dataclass
    class Params(DataclassBase):
        script_path: Text = field(metadata={"help": ""})
        data_files: List[Text] = field(metadata={"help": ""})
        output_path: Text = field(metadata={"help": ""})
        batched: bool = field(default=True, metadata={"help": ""})
        batch_size: int = field(default=128, metadata={"help": ""})
        num_proc: int = field(default=1, metadata={"help": ""})

    def __init__(self, params: Params) -> None:
        self.params = params

    def convert(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs  # use only designated args

        dataset = load_dataset(
            self.params.script_path,
            data_files=self.params.data_files,
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

        self.save_dataset(dataset, self.params.output_path)

    def encode(self, instance: Dict[Text, Any]) -> Dict[Text, Any]:
        result: Union[Text, List[Text]]
        content = instance["content"]
        if isinstance(content, list):
            result = [self.tokenize_text(text) for text in content]
        else:
            result = self.tokenize_text(content)
        return {"output_data": result}

    def save_dataset(
        self, dataset: Union[Dataset, DatasetDict], output_path: Text
    ) -> None:
        data: List[Text] = dataset["output_data"]  # type: ignore
        with open(output_path, mode="w") as stream:
            for group in lazy_groups_of(data, group_size=1000):
                stream.write("\n".join(group))

    @classmethod
    def add_argparse_args(
        cls, parent_parser: DataclassArgumentParser
    ) -> DataclassArgumentParser:
        return DataclassArgumentParser(
            dataclass_types=cls.Params,
            parents=[parent_parser],
            add_help=False,
        )

    def tokenize_text(self, text: Text) -> Text:
        tokens = tokenize_python(text, keep_comments=True)
        return " ".join(tokens)

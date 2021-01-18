import logging
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Text, Type, Union

from datasets import Dataset, DatasetDict, load_dataset
from formerbox.common.dataclass_argparse import MISSING, DataclassBase
from formerbox.common.has_params import HasParsableParams, ParamsType
from formerbox.common.registrable import Registrable
from formerbox.utils import append_path_suffix, lazy_groups_of

Instance = Optional[Text]

logger = logging.getLogger(__name__)


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


class DatasetConverter(Registrable, HasParsableParams[ParamsType], metaclass=ABCMeta):
    params: ParamsType
    params_type: Type[ParamsType]

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


class DatasetConverterBase(DatasetConverter, DatasetProcessingMixin, metaclass=ABCMeta):
    @dataclass
    class Params(DataclassBase):
        script_path: Text = field(
            default=MISSING,
            metadata={
                "help": "The path to the dataset processing script with the dataset builder."
                " Can be either: \n"
                " * a local path to processing script or the directory containing the script"
                " (if the script has the same name as the directory)\n"
                " * a dataset identifier on HuggingFace AWS bucket (list all available"
                " datasets and ids with datasets.list_datasets())."
            },
        )
        data_files: List[Text] = field(
            default_factory=MISSING,
            metadata={"help": "Defining the data_files of the dataset configuration"},
        )
        output_path: Text = field(
            default=MISSING,
            metadata={"help": "The output directory to save converted text datasets."},
        )
        batched: bool = field(
            default=True,
            metadata={
                "help": "Whether or not to provide batches of examples to the function."
                " Default is set to `True`."
            },
        )
        batch_size: int = field(
            default=128,
            metadata={
                "help": "The number of examples per batch provided to function"
                " if batched=True batch_size <= 0 or batch_size == None:"
                " Provide the full dataset as a single batch to function."
                " Default is set to `128`."
            },
        )
        num_proc: int = field(
            default=1,
            metadata={
                "help": "The number of processes for multiprocessing."
                " Default is set to `1`."
            },
        )

    def lookup_data_files(self, data_files: List[Text]) -> List[Text]:
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


@DatasetConverter.register("default", constructor="from_partial")
class DefaultDatasetConverter(DatasetConverterBase):
    @dataclass
    class Params(DatasetConverterBase.Params):
        train_test_split: bool = field(
            default=False,
            metadata={
                "help": "Whether the converted dataset should be splitted into"
                " train/valid/test subsets."
            },
        )
        train_size: float = field(
            default=0.8,
            metadata={
                "help": "The proportion for a training subset of the original dataset."
                " Default is set to `0.8`."
            },
        )
        valid_size: float = field(
            default=0.1,
            metadata={
                "help": "The proportion for a validation subset of the original dataset."
                " Default is set to `0.1`."
            },
        )
        test_size: float = field(
            default=0.1,
            metadata={
                "help": "The proportion for a test subset of the original dataset."
                " Default is set to `0.1`."
            },
        )

    params: Params
    params_type: Type[Params] = Params

    def __init__(self, params: Params) -> None:
        self.params = params

    def convert(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs  # use only designated args
        # search data_files if a user specified a pattern to use
        # otherwise the method will return the input values
        data_files = self.lookup_data_files(self.params.data_files)
        # prepare the output dir for writing results
        Path(self.params.output_path).parent.mkdir(exist_ok=True)

        dataset = load_dataset(
            self.params.script_path,
            data_files=data_files,
            split="train",
        )

        assert isinstance(dataset, Dataset)
        dataset = self.distinct_dataset(
            dataset,
            column="content",
            num_proc=self.params.num_proc,
        )

        assert isinstance(dataset, Dataset)
        dataset = dataset.map(
            self.encode,
            batched=self.params.batched,
            batch_size=self.params.batch_size,
            num_proc=self.params.num_proc,
        )

        # make sure we preserve the columns
        # this is required to avoid issues
        # with splitting dataset using the
        # `train_test_split` method
        dataset.set_format(columns=None)

        train_dataset: Dataset = dataset
        if self.params.train_test_split:
            # sample and save the validation dataset
            if self.params.valid_size > 0:
                dataset_split = train_dataset.train_test_split(
                    test_size=self.params.valid_size
                )

                assert isinstance(dataset_split, DatasetDict)
                train_dataset, valid_dataset = dataset_split.values()
                valid_path = append_path_suffix(self.params.output_path, ".valid")
                self.save_dataset(valid_dataset, valid_path)

            # sample and save the test dataset
            if self.params.test_size > 0:
                dataset_split = train_dataset.train_test_split(
                    test_size=self.params.test_size
                )

                assert isinstance(dataset_split, DatasetDict)
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
            result = [self.preprocess_text(text) for text in content]
        else:
            result = self.preprocess_text(content)
        return {"output_data": result}

    def save_dataset(
        self, dataset: Union[Dataset, DatasetDict], output_path: Text
    ) -> None:
        data: List[Text] = dataset["output_data"]  # type: ignore
        with open(output_path, mode="w") as stream:
            for group in lazy_groups_of(data, group_size=1000):
                instances = [instance for instance in group if instance]
                stream.write("\n".join(instances))

    def preprocess_text(self, text: Text) -> Instance:
        # workaround to avoid disambiguation in parsing text datasets
        return text.replace("\b", "\r")

import json
from abc import abstractmethod
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, List, Optional, Text, Union, Tuple, Iterator

import datasets
import pyarrow as pa
import pyarrow.json as paj


@dataclass
class JsonlinesConfig(datasets.BuilderConfig):
    """BuilderConfig for JSON lines."""

    features: Optional[datasets.Features] = None
    field: Optional[Text] = None
    use_threads: Optional[bool] = True
    block_size: Optional[int] = None
    newlines_in_values: Optional[bool] = True

    @property
    def pa_read_options(self) -> paj.ReadOptions:
        return paj.ReadOptions(use_threads=self.use_threads, block_size=self.block_size)

    @property
    def pa_parse_options(self) -> paj.ParseOptions:
        return paj.ParseOptions(
            explicit_schema=self.schema, newlines_in_values=self.newlines_in_values
        )

    @property
    def schema(self) -> pa.Schema:
        return pa.schema(self.features.type) if self.features is not None else None


class Jsonlines(datasets.ArrowBasedBuilder):
    BUILDER_CONFIG_CLASS = JsonlinesConfig

    def _info(self) -> datasets.DatasetInfo:
        assert isinstance(self.config, self.BUILDER_CONFIG_CLASS)
        return datasets.DatasetInfo(features=self.config.features)

    def _split_generators(
        self, dl_manager: datasets.DownloadManager
    ) -> List[datasets.SplitGenerator]:
        """We handle string, list and dicts in datafiles"""
        if not self.config.data_files:
            raise ValueError(
                f"At least one data file must be specified, "
                f"but got data_files={self.config.data_files}"
            )

        data_files = dl_manager.download_and_extract(self.config.data_files)
        if isinstance(data_files, (str, list, tuple)):
            files = data_files
            if isinstance(files, str):
                files = [files]

            return [
                datasets.SplitGenerator(
                    name=str(datasets.Split.TRAIN), gen_kwargs={"files": files}
                )
            ]

        splits = []
        for split_name, files in data_files.items():
            if isinstance(files, str):
                files = [files]
            splits.append(
                datasets.SplitGenerator(name=split_name, gen_kwargs={"files": files})
            )

        return splits

    def _generate_tables(self, files: List[Text]) -> Iterator[Tuple[int, pa.Table]]:
        assert isinstance(self.config, self.BUILDER_CONFIG_CLASS)
        for i, file in enumerate(files):
            try:
                dataset = self._prepare_dataset(file)
                pa_table = paj.read_json(
                    dataset,
                    read_options=self.config.pa_read_options,
                    parse_options=self.config.pa_parse_options,
                )

            except pa.ArrowInvalid as error:
                raise ValueError(
                    f"Not able to read records in the JSON file at {file}. "
                    f"Your dataset seems to be formatted incorrectly. "
                    f"It should be formatted in jsonlines format."
                ) from error

            yield i, pa_table

    @abstractmethod
    def _generate_examples(self, **kwargs: Dict[Text, Any]) -> Dict[Text, Any]:
        raise NotImplementedError()

    def _prepare_dataset(self, file: Text) -> BytesIO:
        dataset: List[Text] = []
        with open(file) as stream:
            for line in stream.readlines():
                jsonl_line = self._read_jsonline(line)
                dataset.append(json.dumps(jsonl_line))

        return BytesIO("\n".join(dataset).encode("utf-8"))

    def _read_jsonline(self, line: Text) -> Union[Dict[Text, Any], Any]:
        assert isinstance(self.config, self.BUILDER_CONFIG_CLASS)
        decoded_line = json.loads(line, strict=False)
        if self.config.field is not None:
            return decoded_line[self.config.field]
        return decoded_line

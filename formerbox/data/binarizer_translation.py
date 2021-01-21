import logging
from dataclasses import dataclass, field
from itertools import zip_longest
from typing import Any, Optional, Text, Type

from formerbox.common.dataclass_argparse import MISSING
from formerbox.data.binarizer import Binarizer, dataset_dest_filepath
from formerbox.data.binarizer_default import DefaultBinarizer
from formerbox.data.indexed_dataset import IndexedDatasetBuilderBase
from formerbox.data.indexed_dataset_setup import IndexedDatasetSetup
from transformers import PreTrainedTokenizerFast

logger = logging.getLogger(__name__)


# pylint: disable=arguments-differ
@Binarizer.register("translation", constructor="from_partial")
class TranslationBinarizer(DefaultBinarizer):
    @dataclass
    class Params(DefaultBinarizer.Params):
        src_lang: Text = field(
            default=MISSING,
            metadata={
                "help": "A source language extension."
                " Used in mono-bilingual translation tasks."
            },
        )
        tgt_lang: Optional[Text] = field(
            default=None,
            metadata={
                "help": "A target language extension."
                " Used in bilingual translation tasks."
                " Default value is `None`."
            },
        )

    params: Params
    params_type: Type[Params] = Params

    def __init__(
        self,
        dataset_setup: IndexedDatasetSetup,
        tokenizer: PreTrainedTokenizerFast,
        params: Params,
    ) -> None:
        super().__init__(dataset_setup, tokenizer, params)
        self.params = params

    def binarize_dataset(
        self,
        filename: Text,
        output_prefix: Text,
        **kwargs: Any,
    ) -> None:
        del kwargs  # support arbitrary arguments

        # prepare indexed dataset builders for src and tgt languages
        src_dataset_builder = self.src_dataset_builder(output_prefix)
        tgt_dataset_builder = self.tgt_dataset_builder(output_prefix)

        # convert text to ids and write to the data files
        self.binarize(
            filename=filename,
            src_consumer=src_dataset_builder,
            tgt_consumer=tgt_dataset_builder,
        )

        # write meta data and type info
        logger.info("Finilizing the results")

        src_dataset_builder.finalize()
        if tgt_dataset_builder is not None:
            tgt_dataset_builder.finalize()

    def binarize(
        self,
        filename: Text,
        src_consumer: IndexedDatasetBuilderBase,
        tgt_consumer: Optional[IndexedDatasetBuilderBase],
        **kwargs: Any,
    ) -> None:
        del kwargs  # use only designated args

        # process source dataset
        src_filename = f"{filename}.{self.params.src_lang}"
        src_dataset = self.process_dataset(
            src_filename,
            script_path="text",
            script_version=None,
            remove_columns=["text"],
        )

        if self.params.tgt_lang is not None:
            # process target dataset if present
            tgt_filename = f"{filename}.{self.params.tgt_lang}"
            tgt_dataset = self.process_dataset(
                tgt_filename,
                script_path="text",
                script_version=None,
                remove_columns=["text"],
            )

            logger.info("Processing source and target files")
            logger.info("Source file: %s", src_filename)
            logger.info("Target file: %s", tgt_filename)
        else:
            # otherwise use an empty list for zipping
            tgt_dataset = []

            logger.info("Processing only source file: %s", src_filename)

        # zip processed datasets and write instances to consumers
        for src_instance, tgt_instance in zip_longest(src_dataset, tgt_dataset):
            # source instance must exist and be a dictionary
            assert isinstance(src_instance, dict)
            self.write_instance(src_instance, src_consumer)

            # make sure target consumer exists and instance is a dictionary
            if tgt_consumer is not None and isinstance(tgt_instance, dict):
                self.write_instance(tgt_instance, tgt_consumer)

    def dataset_builder(
        self, output_prefix: Text, lang: Text
    ) -> IndexedDatasetBuilderBase:
        filepath_prefix = f"{output_prefix}.{lang}"
        data_filepath = dataset_dest_filepath(filepath_prefix, extension="bin")
        index_filepath = dataset_dest_filepath(filepath_prefix, extension="idx")

        dataset_builder = self.dataset_setup.dataset_builder_type(
            data_filepath=data_filepath,
            index_filepath=index_filepath,
            dtype=self.dataset_setup.dataset_dtype,
            dataset_type=self.dataset_setup.dataset_type,
        )

        return dataset_builder

    def src_dataset_builder(self, output_prefix: Text) -> IndexedDatasetBuilderBase:
        return self.dataset_builder(output_prefix, lang=self.params.src_lang)

    def tgt_dataset_builder(
        self, output_prefix: Text
    ) -> Optional[IndexedDatasetBuilderBase]:
        if self.params.tgt_lang is None:
            return None

        return self.dataset_builder(output_prefix, lang=self.params.tgt_lang)

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Text, Type, Union

from formerbox.data.binarizer import Binarizer, BinarizerBase, dataset_dest_filepath
from formerbox.data.indexed_dataset import IndexedDatasetBuilderBase
from formerbox.data.indexed_dataset_setup import IndexedDatasetSetup
from formerbox.data.tokenizers.tokenization_base import Padding, Truncation
from formerbox.utils import iter_stide
from transformers import PreTrainedTokenizerFast

logger = logging.getLogger(__name__)

# pylint: disable=arguments-differ
@Binarizer.register(name="default", constructor="from_partial")
class DefaultBinarizer(BinarizerBase):
    @dataclass
    class Params(BinarizerBase.Params):
        truncation: Truncation = field(
            default="do_not_truncate",
            metadata={"help": "Activates and controls the truncation."},
        )
        padding: Padding = field(
            default="do_not_pad",
            metadata={"help": "Controls the padding strategy."},
        )
        max_length: Optional[int] = field(
            default=None,
            metadata={
                "help": "Pad to the maximum length specified with this argument."
                " Note, that the default value is tokenizer's `max_model_length`."
                " Changing this argument will override default settings."
                " One might override if tokenizer doesn't specify `max_model_length`."
            },
        )
        stride: int = field(
            default=0,
            metadata={
                "help": "If set to a number along with max_length, "
                " the overflowing tokens returned when return_overflowing_tokens=True"
                " will contain some tokens from the end of the truncated sequence"
                " returned to provide some overlap between truncated and overflowing sequences."
                " The value of this argument defines the number of overlapping tokens."
            },
        )
        return_overflowing_tokens: bool = field(
            default=False,
            metadata={"help": "Whether or not to return overflowing token sequences."},
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

        # prepare indexed dataset builder
        data_filepath = dataset_dest_filepath(output_prefix, extension="bin")
        index_filepath = dataset_dest_filepath(output_prefix, extension="idx")
        dataset_builder = self.dataset_setup.dataset_builder_type(
            data_filepath=data_filepath,
            index_filepath=index_filepath,
            dtype=self.dataset_setup.dataset_dtype,
            dataset_type=self.dataset_setup.dataset_type,
        )

        # convert text to ids and write to the data file
        logger.info("Processing the data file: %s", filename)
        self.binarize(filename=filename, consumer=dataset_builder)

        # write meta data and type info
        logger.info("Finilizing the results")
        dataset_builder.finalize()

    def binarize(
        self,
        filename: Text,
        consumer: IndexedDatasetBuilderBase,
        **kwargs: Any,
    ) -> None:
        del kwargs  # use only designated args

        dataset = self.process_dataset(
            filename,
            script_path="text",
            script_version=None,
            remove_columns=["text"],
        )

        for instance in dataset:
            assert isinstance(instance, dict)
            self.write_instance(instance, consumer)

    def encode(self, instance: Dict[Text, Any]) -> Dict[Text, Any]:
        result: List[Union[int, List[int]]] = []
        lines = instance["text"]

        # use tokenizer associated model max length
        # if max_length isn't set explicitly
        if self.params.max_length is None:
            max_length = getattr(self.tokenizer, "model_max_length", None)
        else:
            max_length = self.params.max_length

            # update tokenizer associated model max length config
            # this is useful since we might wanna save tokenizer
            # to reproduce expected behavior after data preprocessing
            init_kwargs: Dict[Text, Any] = getattr(self.tokenizer, "init_kwargs", {})
            init_kwargs["model_max_length"] = max_length
            setattr(self.tokenizer, "init_kwargs", init_kwargs)

        # make sure we explicitly truncate if max_length is provided
        truncation = self.params.truncation
        if truncation == "do_not_truncate" and max_length is not None:
            truncation = "only_first"  # same as `truncation=True`

        try:
            # split long documents into smaller overlaping chunks
            # this is a workaround to fix tokenizers internal issues
            # with processing long sequences with special tokens
            batch = self.preprocess_batch(lines)
            # tokenize input text and feed to consumer handler
            encoding = self.tokenizer(
                batch,
                truncation=truncation,
                padding=self.params.padding,
                max_length=self.params.max_length,
                stride=self.params.stride,
                return_overflowing_tokens=self.params.return_overflowing_tokens,
            )

            result = encoding.input_ids

        except TypeError as err:
            logger.warning("Unable to tokenize a text, error: %s", err)

        return {"input_ids": result}

    def preprocess_batch(self, batch: Union[Text, List[Text]]) -> List[Text]:
        if not isinstance(batch, list):
            batch = [batch]

        result = []
        for instance in batch:
            sentences = iter_stide(instance.split(" "), chunk_size=512, stride=32)
            for sentence in sentences:
                result.append(" ".join(sentence))

        return result

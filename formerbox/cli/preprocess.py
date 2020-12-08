import logging
import os
from argparse import Namespace, _SubParsersAction
from dataclasses import dataclass, field
from time import time
from typing import Any, Dict, Optional, Text, Tuple, Union

from formerbox.cli.functional import temp_filepath
from formerbox.cli.subcommand import Subcommand
from formerbox.common.dataclass_argparse import (
    MISSING,
    DataclassArgumentParser,
    DataclassBase,
    get_params_item,
)
from formerbox.data import TokenizerBase
from formerbox.data.binarizer import Binarizer
from formerbox.data.indexed_dataset_setup import IndexedDatasetSetup
from transformers import PreTrainedTokenizerFast as Tokenizer

logger = logging.getLogger(__name__)


@Subcommand.register("preprocess")
class Preprocess(Subcommand):
    @dataclass
    class Params(DataclassBase):
        tokenizer: Text = field(
            default=MISSING,
            metadata={
                "choices": TokenizerBase,
                "help": "The name of a tokenizer to load from.",
            },
        )
        tokenizer_path: Text = field(
            default=MISSING,
            metadata={"help": "A path to the pre-trained tokenizer files."},
        )
        binarizer: Text = field(
            default=MISSING,
            metadata={
                "choices": Binarizer,
                "help": "The name of a registered binarizer to use.",
            },
        )
        output_path: Text = field(
            default=MISSING,
            metadata={"help": "An output path for writing output files to."},
        )
        train_prefix: Text = field(
            default=MISSING,
            metadata={"help": "Train dataset text file prefix."},
        )
        valid_prefix: Optional[Text] = field(
            default=None,
            metadata={"help": "Validation dataset text file prefix."},
        )
        test_prefix: Optional[Text] = field(
            default=None,
            metadata={"help": "Test dataset text file prefix."},
        )
        legacy_format: bool = field(
            default=False,
            metadata={
                "help": "Whether to save the tokenizer in legacy format,"
                " i.e. with tokenizer specific vocabulary and separate added_tokens files"
                " or in the unified JSON file format of the `tokenizers` library (default)."
            },
        )

    def add_subparser(
        self, parser: _SubParsersAction
    ) -> Tuple[DataclassArgumentParser, Dict[Text, Any]]:
        description = """Preprocess the text dataset into indexed datasets."""
        subparser = self._add_parser(
            parser,
            name=self.name,
            description=description,
            help=description,
        )

        # add command arguments
        subparser.add_arguments(self.Params)
        # add index setup arguments
        IndexedDatasetSetup.add_argparse_params(subparser)

        def add_dynamic_args(parser: DataclassArgumentParser) -> None:
            # get the parsed command arguments
            parsed_params = subparser.parse_args_into_dataclasses(
                return_remaining_strings=True
            )
            params = parsed_params[0]
            assert isinstance(params, self.Params)

            # add dybamic args to the subparser
            binarizer_cls, _ = Binarizer.from_registry(params.binarizer)
            binarizer_cls.add_argparse_params(subparser)

            # inject dataclass_types to the parent parser
            parser.dataclass_types = subparser.dataclass_types

        defaults = dict(
            func=preprocess,
            add_dynamic_args=add_dynamic_args,
        )

        return subparser, defaults


def save_tokenizer(
    tokenizer: Tokenizer, output_path: Text, legacy_format: bool
) -> None:
    # prepare the tokenizer path
    output_path = os.path.join(output_path, "tokenizer")

    # keep only token-related items in the config
    token_config_kwargs = getattr(tokenizer, "init_kwargs", {})
    token_config_kwargs.pop("name_or_path", None)
    token_config_kwargs.pop("special_tokens_map_file", None)

    setattr(tokenizer, "init_kwargs", token_config_kwargs)

    # save the pre-trained tokenizer
    tokenizer.save_pretrained(
        save_directory=output_path,
        legacy_format=legacy_format,
    )


def preprocess(params: Tuple[Union[DataclassBase, Namespace], ...]) -> None:
    # pylint: disable=too-many-locals
    # make sure tokenizer parallelizm is disabled
    # since it might cause deadlocks while preprocessing
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    cmd_params = get_params_item(params, params_type=Preprocess.Params)

    # prepare the pretrained tokenizer
    tokenizer_cls = TokenizerBase.from_name(cmd_params.tokenizer)
    tokenizer = tokenizer_cls.from_pretrained(cmd_params.tokenizer_path)

    # prepare the dataset setup
    dataset_setup_params = get_params_item(
        params, params_type=IndexedDatasetSetup.Params
    )

    assert dataset_setup_params is not None
    dataset_setup = IndexedDatasetSetup.from_args(dataset_setup_params)

    # prepare the dataset binarizer
    binarizer_cls, binarizer_init = Binarizer.from_registry(cmd_params.binarizer)
    binarizer_params = get_params_item(params, params_type=binarizer_cls.params_type)
    binarizer = binarizer_init(
        dataset_setup=dataset_setup,
        tokenizer=tokenizer,
        params=binarizer_params,
    )

    # prepare the output dir for writing data files
    os.makedirs(cmd_params.output_path, exist_ok=True)

    # run dataset binarization for each split
    for split, datafile_prefix in (
        ("train", cmd_params.train_prefix),
        ("valid", cmd_params.valid_prefix),
        ("test", cmd_params.test_prefix),
    ):
        # skip empty dataset splits
        if datafile_prefix is None:
            continue

        logger.info("Start processing %s subset...", split)
        start_time = time()

        output_prefix = temp_filepath(
            filepath=datafile_prefix,
            suffix="",
            output_path=cmd_params.output_path,
        )

        binarizer.binarize_dataset(
            filename=datafile_prefix,
            output_prefix=output_prefix,
        )

        time_delta = time() - start_time
        logger.info("Wall time: %.3fs", time_delta)

    # save the pretrained tokenizer to the output directory
    # we do this after data preprocessing because some updates
    # might happen on the tokenizer state during the preprocessing
    save_tokenizer(
        tokenizer,
        output_path=cmd_params.output_path,
        legacy_format=cmd_params.legacy_format,
    )

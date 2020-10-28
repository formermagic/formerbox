import logging
import os
from argparse import Namespace, _SubParsersAction
from dataclasses import dataclass, field
from time import time
from typing import Any, Dict, Optional, Text, Tuple, Union

from formerbox.cli.functional import temp_filepath
from formerbox.cli.subcommand import Subcommand
from formerbox.common.dataclass_argparse import (
    DataclassArgumentParser,
    DataclassBase,
    get_params_item,
)
from formerbox.data.binarizer import Binarizer
from formerbox.data.indexed_dataset_setup import IndexedDatasetSetup
from formerbox.modules import TokenizerModule
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

logger = logging.getLogger(__name__)


@Subcommand.register("preprocess")
class Preprocess(Subcommand):
    @dataclass
    class Params(DataclassBase):
        tokenizer: Text = field(
            metadata={
                "choices": TokenizerModule,
                "help": "The name of a registered tokenizer module to use.",
            },
        )
        binarizer: Text = field(
            metadata={
                "choices": Binarizer,
                "help": "The name of a registered binarizer to use.",
            },
        )
        output_path: Text = field(
            metadata={"help": "An output path for writing output files to."},
        )
        train_prefix: Text = field(
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
            tokenizer_cls, _ = TokenizerModule.from_registry(params.tokenizer)
            tokenizer_cls.add_argparse_params(subparser)
            binarizer_cls, _ = Binarizer.from_registry(params.binarizer)
            binarizer_cls.add_argparse_params(subparser)

            # inject dataclass_types to the parent parser
            parser.dataclass_types = subparser.dataclass_types

        defaults = dict(
            func=preprocess,
            add_dynamic_args=add_dynamic_args,
        )

        return subparser, defaults


def save_tokenizer(tokenizer: Tokenizer, output_path: Text) -> None:
    output_path = os.path.join(output_path, "tokenizer")

    # keep only token-related items in the config
    token_config_kwargs = {}
    init_kwargs = getattr(tokenizer, "init_kwargs", {})
    for arg_name, arg_value in init_kwargs.items():
        if "file" in arg_name:
            continue
        if arg_value is None:
            continue

        token_config_kwargs[arg_name] = arg_value

    setattr(tokenizer, "init_kwargs", token_config_kwargs)
    tokenizer.save_pretrained(save_directory=output_path)


def preprocess(params: Tuple[Union[DataclassBase, Namespace], ...]) -> None:
    # pylint: disable=too-many-locals
    # make sure tokenizer parallelizm is disabled
    # since it might cause deadlocks while preprocessing
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    cmd_params = get_params_item(params, params_type=Preprocess.Params)

    # prepare the pretrained tokenizer
    tokenizer_cls, _ = TokenizerModule.from_registry(cmd_params.tokenizer)
    tokenizer_params = get_params_item(params, params_type=tokenizer_cls.params_type)
    tokenizer = tokenizer_cls.from_pretrained(params=tokenizer_params)

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
    # save the pretrained tokenizer to the output directory
    save_tokenizer(tokenizer, cmd_params.output_path)

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

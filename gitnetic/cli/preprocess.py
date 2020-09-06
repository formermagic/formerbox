"""Preprocess text datasets represented as strings separated by newline.
The script will output two files – an index file (.idx) and a data file (.bin).
Files can be read by `IndexedDataset` and its subclasses.

Preprocessed data file contains binarized texts as token ids. Index file
contains meta information about binarized data (e.g. step size, total length, etc).
"""
import logging
import os
import time
from argparse import ArgumentParser
from multiprocessing import Pool
from typing import Any, Callable, Optional, Text

from tqdm import tqdm
from transformers import PreTrainedTokenizerFast

from gitnetic.data import Binarizer, dataset_dest_filepath, find_offsets
from gitnetic.data.indexed_dataset_setup import IndexedDatasetSetup
from gitnetic.tasks.base_transformers import TokenizerModule

logger = logging.getLogger(__name__)


def temp_filepath(filepath: Text, suffix: Text, output_path: Text) -> Text:
    filename = os.path.basename(filepath)
    output_prefix = os.path.join(output_path, f"{filename}{suffix}")
    return output_prefix


# pylint: disable=too-many-arguments, too-many-locals
def preprocess(
    make_binarizer: Callable[[], Binarizer],
    train_prefix: Text,
    valid_prefix: Optional[Text],
    test_prefix: Optional[Text],
    output_path: Text,
    num_workers: int,
    dataset_impl: Text,
    **kwargs: Any,
) -> None:
    os.makedirs(output_path, exist_ok=True)

    for filepath in [train_prefix, valid_prefix, test_prefix]:
        if filepath is None:
            continue

        # split file into chunks for processing in parallel
        _, offsets = find_offsets(filepath, num_chunks=num_workers)
        logger.info("Preprocessing %s file...", filepath)
        start_time = time.time()

        # binarize file with num_workers processes
        with tqdm(total=num_workers) as pbar:
            pool = Pool(processes=num_workers)
            for worker_idx in range(num_workers):
                output_prefix = temp_filepath(filepath, str(worker_idx), output_path)

                binarizer = make_binarizer()

                pool.apply_async(
                    binarizer.binarize_dataset,
                    args=(
                        filepath,
                        output_prefix,
                        offsets[worker_idx],
                        offsets[worker_idx + 1],
                    ),
                    kwds=kwargs,
                    callback=lambda _: pbar.update(),  # pylint: disable=cell-var-from-loop
                )

            pool.close()
            pool.join()

        # prepare dest paths for merged files
        output_prefix = temp_filepath(filepath, "", output_path)
        data_filepath = dataset_dest_filepath(output_prefix, extension="bin")
        index_filepath = dataset_dest_filepath(output_prefix, extension="idx")

        # prepare a dataset builder instance
        dataset_setup = IndexedDatasetSetup.from_args(dataset_impl)
        dataset_builder = dataset_setup.dataset_builder_type(
            data_filepath,
            index_filepath,
            dataset_setup.dataset_dtype,
            dataset_setup.dataset_type,
        )

        # merge temp files contents and remove files from disk
        for worker_idx in range(num_workers):
            output_prefix = temp_filepath(filepath, str(worker_idx), output_path)
            temp_file_path = dataset_dest_filepath(output_prefix, extension="")
            dataset_builder.merge_file(temp_file_path)

        # write final meta and type data
        dataset_builder.finalize()

        # log execution wall time
        time_delta = time.time() - start_time
        logger.info("Wall time: %.3fs", time_delta)


def make_parser() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    # fmt: off
    parser.add_argument("--train_prefix", type=str, required=True,
                        help="Train dataset text file prefix.")
    parser.add_argument("--valid_prefix", type=str, default=None,
                        help="Validation dataset text file prefix (optional).")
    parser.add_argument("--test_prefix", type=str, default=None,
                        help="Test dataset text file prefix (optional).")
    parser.add_argument("--tokenizer_type", type=str, required=True,
                        help="")
    parser.add_argument("--binarizer_type", type=str, required=True,
                        help="")
    parser.add_argument("--output_path", type=str, required=True,
                        help="An output path for writing output files to.")
    parser.add_argument("--num_workers", type=int, required=True,
                        help="A number of processes to perform actions in parallel.")
    # fmt: on

    # add indexed dataset impl argument
    IndexedDatasetSetup.add_arguments(parser)

    return parser


def main() -> None:
    # parse args for basic preprocessing
    parser = make_parser()
    args = vars(parser.parse_known_args()[0])

    # prepare args for loading a pre-trained tokenizer
    tokenizer_cls, _ = TokenizerModule.from_registry(args["tokenizer_type"])
    parser = tokenizer_cls.add_argparse_args(parser, stage="tokenize")
    # prepare args for building a binarizer
    binarizer_cls, _ = Binarizer.from_registry(args["binarizer_type"])
    parser = binarizer_cls.add_argparse_args(parser)

    args = vars(parser.parse_known_args()[0])

    # build the selected pre-trained tokenizer
    tokenizer_path = args["tokenizer_path"]
    tokenizer = tokenizer_cls.from_pretrained(tokenizer_path, **args)

    # prepare a binarizer builder
    def make_binarizer() -> Binarizer:
        # prepare the selected dataset setup
        dataset_impl = args["dataset_impl"]
        dataset_setup = IndexedDatasetSetup.from_args(dataset_impl)
        # build a binarizer with the selected dataset setup and tokenizer
        assert isinstance(tokenizer, PreTrainedTokenizerFast)
        binarizer = binarizer_cls(dataset_setup, tokenizer)
        return binarizer

    # preprocess inputs with added args, selected tokenizer and binarizer
    preprocess(make_binarizer, **args)


if __name__ == "__main__":
    main()

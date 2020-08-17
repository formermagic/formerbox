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
from typing import Optional, Text

from tqdm import tqdm
from transformers import PreTrainedTokenizerFast

from gitnetic.data import (
    Binarizer,
    IndexedDatasetBuilder,
    dataset_dest_filepath,
    find_offsets,
)
from gitnetic.tasks.codebert import CodeBertTokenizerFast

logger = logging.getLogger(__name__)


def temp_filepath(filepath: Text, suffix: Text, output_path: Text) -> Text:
    filename = os.path.basename(filepath)
    output_prefix = os.path.join(output_path, f"{filename}{suffix}")
    return output_prefix


def load_tokenizer(
    tokenizer_path: Text,
    tokenizer_add_prefix_space: bool,
    tokenizer_trim_offsets: bool,
    tokenizer_lowercase: bool,
) -> PreTrainedTokenizerFast:
    tokenizer = CodeBertTokenizerFast.from_pretrained(
        pretrained_model_name_or_path=tokenizer_path,
        add_prefix_space=tokenizer_add_prefix_space,
        trim_offsets=tokenizer_trim_offsets,
        lowercase=tokenizer_lowercase,
    )

    assert isinstance(
        tokenizer, PreTrainedTokenizerFast
    ), "Tokenizer must be a subclass of PreTrainedTokenizerFast."

    return tokenizer


# pylint: disable=too-many-arguments, too-many-locals
def preprocess(
    train_prefix: Text,
    valid_prefix: Optional[Text],
    test_prefix: Optional[Text],
    tokenizer_path: Text,
    tokenizer_add_prefix_space: bool,
    tokenizer_trim_offsets: bool,
    tokenizer_lowercase: bool,
    tokenizer_max_length: int,
    output_path: Text,
    num_workers: int,
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

                tokenizer = load_tokenizer(
                    tokenizer_path,
                    tokenizer_add_prefix_space,
                    tokenizer_trim_offsets,
                    tokenizer_lowercase,
                )

                binarizer = Binarizer(
                    tokenizer=tokenizer, tokenizer_max_length=tokenizer_max_length,
                )

                pool.apply_async(
                    binarizer.binarize_dataset,
                    args=(
                        filepath,
                        output_prefix,
                        offsets[worker_idx],
                        offsets[worker_idx + 1],
                    ),
                    callback=lambda _: pbar.update(),  # pylint: disable=cell-var-from-loop
                )

            pool.close()
            pool.join()

        # prepare dest paths for merged files
        output_prefix = temp_filepath(filepath, "", output_path)
        data_filepath = dataset_dest_filepath(output_prefix, extension="bin")
        index_filepath = dataset_dest_filepath(output_prefix, extension="idx")

        # merge temp files contents and remove files from disk
        dataset_builder = IndexedDatasetBuilder(data_filepath)
        with dataset_builder:
            for worker_idx in range(num_workers):
                output_prefix = temp_filepath(filepath, str(worker_idx), output_path)
                temp_file_path = dataset_dest_filepath(output_prefix, extension="")
                dataset_builder.merge_file(temp_file_path)

        # write final meta and type data
        dataset_builder.finalize(index_filepath)

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
    parser.add_argument("--tokenizer_path", type=str, required=True,
                        help="A path to pretrained tokenizer files.")
    parser.add_argument("--tokenizer_add_prefix_space", type=bool, default=False,
                        help="Whether to add a leading space to the first word.")
    parser.add_argument("--tokenizer_trim_offsets", type=bool, default=True,
                        help="Whether the post processing step should trim "
                        "offsets to avoid including whitespaces.")
    parser.add_argument("--tokenizer_lowercase", type=bool, default=True,
                        help="Whether to preprocess text as lowercase.")
    parser.add_argument("--tokenizer_max_length", type=int, default=512,
                        help="A maximum length of text sequence to encode.")
    parser.add_argument("--output_path", type=str, required=True,
                        help="An output path for writing output files to.")
    parser.add_argument("--num_workers", type=int, required=True,
                        help="A number of processes to perform actions in parallel.")
    # fmt: on

    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    preprocess(**vars(args))


if __name__ == "__main__":
    main()
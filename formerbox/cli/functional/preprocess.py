"""Preprocess text datasets represented as strings separated by newline.
The script will output two files – an index file (.idx) and a data file (.bin).
Files can be read by `IndexedDataset` and its subclasses.

Preprocessed data file contains binarized texts as token ids. Index file
contains meta information about binarized data (e.g. step size, total length, etc).
"""
import logging
import os
import time
from multiprocessing import Pool
from typing import Any, Callable, Optional, Text

from formerbox.data import Binarizer
from formerbox.data.binarizer import dataset_dest_filepath, find_offsets
from formerbox.data.indexed_dataset_setup import IndexedDatasetSetup
from tqdm import tqdm

logger = logging.getLogger(__name__)


def temp_filepath(filepath: Text, suffix: Text, output_path: Text) -> Text:
    filename = os.path.basename(filepath)
    output_prefix = os.path.join(output_path, f"{filename}{suffix}")
    return output_prefix


def tqdm_callback(pbar: tqdm) -> Callable[[Any], None]:
    def update(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        pbar.update()

    return update


# pylint: disable=too-many-arguments, too-many-locals
def preprocess(
    binarizer: Binarizer,
    train_prefix: Text,
    valid_prefix: Optional[Text],
    test_prefix: Optional[Text],
    output_path: Text,
    num_workers: int,
    dataset_setup: IndexedDatasetSetup,
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

                pool.apply_async(
                    binarizer.binarize_dataset,
                    args=(
                        filepath,
                        output_prefix,
                        offsets[worker_idx],
                        offsets[worker_idx + 1],
                    ),
                    kwds=kwargs,
                    callback=tqdm_callback(pbar),
                )

            pool.close()
            pool.join()

        # prepare dest paths for merged files
        output_prefix = temp_filepath(filepath, "", output_path)
        data_filepath = dataset_dest_filepath(output_prefix, extension="bin")
        index_filepath = dataset_dest_filepath(output_prefix, extension="idx")

        # prepare a dataset builder instance
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

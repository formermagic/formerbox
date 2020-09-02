import gzip
import json
import os
import shutil
from argparse import ArgumentParser, Namespace
from multiprocessing import Pool
from pathlib import Path
from typing import List, Text, Union

from gitnetic.utils import lazy_groups_of, str2bool
from gitnetic.utils.code_tokenizer import tokenize_python


def read_data(filepath: Union[Text, Path]) -> List[Text]:
    with gzip.open(filepath) as stream:
        jsonlines = stream.readlines()
    parsed_jsonl = [json.loads(jsonl) for jsonl in jsonlines]

    lines: List[Text] = []
    for jsonl in parsed_jsonl:
        try:
            lines.append(jsonl["content"])
        except KeyError:
            continue

    return lines


def pretokenize_line(line: Text, keep_comments: bool) -> Text:
    tokens = tokenize_python(line, keep_comments)
    return " ".join(tokens)


def pretokenize(
    lines: List[Text],
    output_filepath: Union[Text, Path],
    keep_comments: bool,
) -> None:
    with open(output_filepath, mode="w", buffering=1) as stream:
        buffer_size = 128
        for lines_buffer in lazy_groups_of(lines, buffer_size):
            tokenized_lines = [
                pretokenize_line(line, keep_comments) for line in lines_buffer
            ]

            stream.write("\n".join(tokenized_lines))


def merge_files(
    files: List[Union[Text, Path]],
    output_filepath: Union[Text, Path],
    should_remove: bool,
) -> None:
    with open(output_filepath, mode="w") as output:
        for filepath in files:
            with open(filepath, mode="r") as tmp:
                shutil.copyfileobj(tmp, output)
            if should_remove:
                os.remove(filepath)


def pretokenize_file(
    filepath: Union[Text, Path],
    output_filepath: Union[Text, Path],
    num_workers: int,
    keep_comments: bool,
    remove_temp_files: bool,
) -> None:
    if isinstance(filepath, str):
        output_filepath = Path(filepath)
    assert isinstance(output_filepath, Path)

    # prepare chunks of a file to process concurrently
    lines = read_data(filepath)[:1000]
    group_size = len(lines) // num_workers
    groups_of_lines = lazy_groups_of(lines, group_size)

    pool = Pool(processes=num_workers)

    # schedule processing of each chunk
    # this will result in multiple temp files
    worker_files: List[Union[Text, Path]] = []
    for worker_idx, group_of_lines in enumerate(groups_of_lines):
        worker_filepath = output_filepath.parent / f"tempfile_{worker_idx}"
        worker_files.append(worker_filepath)

        pool.apply_async(
            pretokenize, args=(group_of_lines, worker_filepath, keep_comments)
        )

    pool.close()
    pool.join()

    # merge temp files into one output file
    merge_files(
        worker_files,
        output_filepath=output_filepath,
        should_remove=remove_temp_files,
    )


def parse_args() -> Namespace:
    parser = ArgumentParser()
    # fmt: off
    parser.add_argument("--input_files", type=str, nargs="+", default=None, required=True,
                        help="A list of input files to tokenize.")
    parser.add_argument("--output_filepath", type=str, default=None, required=True,
                        help="A path to the output file with tokenized lines.")
    parser.add_argument("--num_workers", type=int, default=None, required=True,
                        help="A number of parallel workers.")
    parser.add_argument("--keep_comments", type=str2bool, default=True, required=False,
                        help="Whether should keep or remove comments from code.")
    parser.add_argument("--remove_temp_files", type=str2bool, default=True, required=False,
                        help="Whether should keep or remove temp files.")
    # fmt: on

    return parser.parse_args()


def main() -> None:
    # parse the arguments
    args = parse_args()

    # prepare required arguments for tokenization
    input_files = [Path(path) for path in args.input_files]
    output_filepath = Path(args.output_filepath)
    num_workers = args.num_workers
    keep_comments = args.keep_comments
    remove_temp_files = args.remove_temp_files

    # prepare the output directory
    try:
        shutil.rmtree(output_filepath.parent)
    except FileNotFoundError:
        pass
    output_filepath.parent.mkdir(exist_ok=True)

    # process each file in chunks parallely
    # this will result in multiple merged temp files
    temp_files: List[Union[Text, Path]] = []
    for idx, filepath in enumerate(input_files):
        temp_filepath = output_filepath.parent / f"temp_merged_{idx}"
        temp_files.append(temp_filepath)

        pretokenize_file(
            filepath,
            temp_filepath,
            num_workers,
            keep_comments,
            remove_temp_files,
        )

    # merge temp files into one output file
    merge_files(
        temp_files,
        output_filepath=output_filepath,
        should_remove=remove_temp_files,
    )


if __name__ == "__main__":
    main()

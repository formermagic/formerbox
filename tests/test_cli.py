import logging
import os
import shlex
import shutil
import sys
from pathlib import Path
from typing import Text

import pytest
from formerbox.cli import main
from formerbox.utils import append_path_suffix

logger = logging.getLogger(__name__)

FIXTURES_DIR_ERROR = "Fixtures dir doesn't exist"


@pytest.mark.parametrize("converter", ["default"])
def test_cli_convert_dataset(
    tmp_path: Path,
    fixtures_directory: Path,
    converter: Text,
) -> None:
    assert os.path.exists(fixtures_directory), FIXTURES_DIR_ERROR
    data_files = fixtures_directory / "tiny_dataset" / "tiny_dataset.jsonl"
    output_path = tmp_path / "tiny_dataset" / "tiny_dataset.src"

    argv = shlex.split(
        f"""
        formerbox-cli convert_dataset       \
            --converter {converter}         \
            --script_path json              \
            --data_files {data_files}       \
            --output_path {output_path}     \
            --batch_size 128                \
            --num_proc 8                    \
            --train_test_split true
        """
    )

    sys.argv = argv
    main(prog="tests")

    assert os.path.exists(append_path_suffix(output_path, ".train"))
    assert os.path.exists(append_path_suffix(output_path, ".valid"))
    assert os.path.exists(append_path_suffix(output_path, ".test"))

    shutil.rmtree(tmp_path)

    assert not os.path.exists(tmp_path)


@pytest.mark.parametrize("tokenizer", ["code_roberta"])
def test_cli_train_tokenizer(
    tmp_path: Path,
    fixtures_directory: Path,
    tokenizer: Text,
) -> None:
    assert os.path.exists(fixtures_directory), FIXTURES_DIR_ERROR
    files = fixtures_directory / "tiny_dataset" / "tiny_dataset.train.src"
    save_directory = tmp_path / "tiny_dataset" / "tokenizer"
    assert not os.path.exists(save_directory)

    save_directory.mkdir(parents=True, exist_ok=True)

    argv = shlex.split(
        f"""
        formerbox-cli train_tokenizer           \
            --tokenizer {tokenizer}             \
            --save_directory {save_directory}   \
            --files {files}                     \
            --vocab_size 20000                  \
            --legacy_format false
        """
    )

    sys.argv = argv
    main(prog="tests")

    assert os.path.exists(save_directory)
    assert len(os.listdir(save_directory)) > 0

    shutil.rmtree(tmp_path)

    assert not os.path.exists(tmp_path)


@pytest.mark.parametrize("tokenizer", ["code_roberta"])
@pytest.mark.parametrize("binarizer", ["default"])
def test_cli_preprocess(
    tmp_path: Path,
    fixtures_directory: Path,
    tokenizer: Text,
    binarizer: Text,
) -> None:
    assert os.path.exists(fixtures_directory), FIXTURES_DIR_ERROR
    tiny_dataset = fixtures_directory / "tiny_dataset" / "tiny_dataset.src"
    train_prefix = append_path_suffix(tiny_dataset, ".train")
    valid_prefix = append_path_suffix(tiny_dataset, ".valid")
    test_prefix = append_path_suffix(tiny_dataset, ".test")

    tokenizer_path = fixtures_directory / "tiny_dataset" / "tokenizer"
    output_path = tmp_path / "tiny_dataset.bin"
    assert not os.path.exists(output_path)

    argv = shlex.split(
        f"""
        formerbox-cli preprocess                \
            --train_prefix {train_prefix}       \
            --valid_prefix {valid_prefix}       \
            --test_prefix {test_prefix}         \
            --tokenizer {tokenizer}             \
            --tokenizer_path {tokenizer_path}   \
            --legacy_format false               \
            --binarizer {binarizer}             \
            --max_length 512                    \
            --return_overflowing_tokens true    \
            --output_path {output_path}         \
            --num_proc 8                        \
            --batch_size 512                    \
            --batched true                      \
            --dataset_impl mmap
        """
    )

    sys.argv = argv
    main(prog="tests")

    assert os.path.exists(output_path)
    assert len(os.listdir(output_path)) >= 3 * 2
    assert len(os.listdir(output_path / "tokenizer")) > 0

    shutil.rmtree(tmp_path)

    assert not os.path.exists(tmp_path)


@pytest.mark.parametrize("task", ["masked_lm"])
def test_cli_train(
    tmp_path: Path,
    fixtures_directory: Path,
    task: Text,
) -> None:
    assert os.path.exists(fixtures_directory), FIXTURES_DIR_ERROR
    config_path = fixtures_directory / "tiny_roberta.yml"
    tiny_dataset = fixtures_directory / "tiny_dataset.bin"

    tokenizer_path = tiny_dataset / "tokenizer"
    train_data_prefix = tiny_dataset / "tiny_dataset.train.src"
    val_data_prefix = tiny_dataset / "tiny_dataset.valid.src"

    save_dir = tmp_path / "pl_checkpoints"

    argv = shlex.split(
        f"""
        formerbox-cli train                             \
            --task {task}                               \
            --config_path {config_path}                 \
            --tokenizer_path {tokenizer_path}           \
                                                        \
            --warmup_steps 2000                         \
            --learning_rate 5e-4                        \
            --power 1.0                                 \
                                                        \
            --train_data_prefix {train_data_prefix}     \
            --val_data_prefix {val_data_prefix}         \
            --max_tokens 1024                           \
            --num_workers 8                             \
                                                        \
            --gpus 0                                    \
            --max_steps 2                               \
            --val_check_interval 10                     \
            --save_step_frequency 10                    \
            --save_dir {save_dir}                       \
            --progress_bar_refresh_rate 0               \
            --log_every_n_steps 1
        """
    )

    sys.argv = argv
    main(prog="tests")

    shutil.rmtree(tmp_path)

    assert not os.path.exists(tmp_path)

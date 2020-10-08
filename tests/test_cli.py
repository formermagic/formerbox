import logging
import os
import shlex
import shutil
import sys
from pathlib import Path
from typing import Text

import pytest
from gitnetic.cli import main
from gitnetic.utils import append_path_suffix

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("converter", ["code-lm-converter"])
def test_cli_convert_dataset(
    tmp_path: Path,
    fixtures_directory: Path,
    converter: Text,
) -> None:
    data_files = fixtures_directory / "tiny_dataset" / "tiny_raw_dataset.jsonl"
    output_path = tmp_path / "tiny_dataset" / "tiny_dataset.src"

    argv = shlex.split(
        f"""
        gitnetic-cli convert_dataset        \
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


@pytest.mark.parametrize("tokenizer", ["transformer-tokenizer-fast"])
def test_cli_train_tokenizer(
    tmp_path: Path,
    fixtures_directory: Path,
    tokenizer: Text,
) -> None:
    files = fixtures_directory / "tiny_dataset" / "tiny_dataset.train.src"
    tokenizer_path = tmp_path / "tiny_dataset" / "tokenizer"
    assert not os.path.exists(tokenizer_path)

    tokenizer_path.mkdir(parents=True, exist_ok=True)

    argv = shlex.split(
        f"""
        gitnetic-cli train_tokenizer            \
            --tokenizer {tokenizer}             \
            --tokenizer_path {tokenizer_path}   \
            --files {files}                     \
            --vocab_size 20000
        """
    )

    sys.argv = argv
    main(prog="tests")

    assert os.path.exists(tokenizer_path)
    assert len(os.listdir(tokenizer_path)) > 0

    shutil.rmtree(tmp_path)

    assert not os.path.exists(tmp_path)


@pytest.mark.parametrize("tokenizer", ["transformer-tokenizer-fast"])
@pytest.mark.parametrize("binarizer", ["flat-binarizer"])
def test_cli_preprocess(
    tmp_path: Path,
    fixtures_directory: Path,
    tokenizer: Text,
    binarizer: Text,
) -> None:
    tiny_dataset = fixtures_directory / "tiny_dataset" / "tiny_dataset.src"
    train_prefix = append_path_suffix(tiny_dataset, ".train")
    valid_prefix = append_path_suffix(tiny_dataset, ".valid")
    test_prefix = append_path_suffix(tiny_dataset, ".test")

    tokenizer_path = fixtures_directory / "tiny_dataset" / "tokenizer"
    output_path = tmp_path / "tiny_dataset.bin"
    assert not os.path.exists(output_path)

    argv = shlex.split(
        f"""
        gitnetic-cli preprocess                 \
            --train_prefix {train_prefix}       \
            --valid_prefix {valid_prefix}       \
            --test_prefix {test_prefix}         \
            --tokenizer {tokenizer}             \
            --tokenizer_path {tokenizer_path}   \
            --binarizer {binarizer}             \
            --max_length 512                    \
            --return_overflowing_tokens True    \
            --output_path {output_path}         \
            --num_proc 8                        \
            --batch_size 512                    \
            --batched True                      \
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


@pytest.mark.parametrize("task", ["transformer-task"])
def test_cli_train(
    tmp_path: Path,
    fixtures_directory: Path,
    task: Text,
) -> None:
    config_path = fixtures_directory / "model-config.yml"
    tiny_dataset = fixtures_directory / "tiny_dataset.bin"

    tokenizer_path = tiny_dataset / "tokenizer"
    train_data_prefix = tiny_dataset / "tiny_dataset.train.src"
    val_data_prefix = tiny_dataset / "tiny_dataset.valid.src"

    save_dir = tmp_path / "pl_checkpoints"

    argv = shlex.split(
        f"""
        gitnetic-cli train                              \
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

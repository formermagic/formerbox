"""
This type stub file was generated by pyright.
"""

import unittest
import torch
from .file_utils import _torch_available

SMALL_MODEL_IDENTIFIER = "julien-c/bert-xsmall-dummy"
DUMMY_UNKWOWN_IDENTIFIER = "julien-c/dummy-unknown"
DUMMY_DIFF_TOKENIZER_IDENTIFIER = "julien-c/dummy-diff-tokenizer"
def parse_flag_from_env(key, default=...):
    ...

def parse_int_from_env(key, default=...):
    ...

_run_slow_tests = parse_flag_from_env("RUN_SLOW", default=False)
_run_custom_tokenizers = parse_flag_from_env("RUN_CUSTOM_TOKENIZERS", default=False)
_tf_gpu_memory_limit = parse_int_from_env("TF_GPU_MEMORY_LIMIT", default=None)
def slow(test_case):
    """
    Decorator marking a test as slow.

    Slow tests are skipped by default. Set the RUN_SLOW environment variable
    to a truthy value to run them.

    """
    ...

def custom_tokenizers(test_case):
    """
    Decorator marking a test for a custom tokenizer.

    Custom tokenizers require additional dependencies, and are skipped
    by default. Set the RUN_CUSTOM_TOKENIZERS environment variable
    to a truthy value to run them.
    """
    ...

def require_torch(test_case):
    """
    Decorator marking a test that requires PyTorch.

    These tests are skipped when PyTorch isn't installed.

    """
    ...

def require_tf(test_case):
    """
    Decorator marking a test that requires TensorFlow.

    These tests are skipped when TensorFlow isn't installed.

    """
    ...

def require_flax(test_case):
    """
    Decorator marking a test that requires JAX & Flax

    These tests are skipped when one / both are not installed

    """
    ...

def require_sentencepiece(test_case):
    """
    Decorator marking a test that requires SentencePiece.

    These tests are skipped when SentencePiece isn't installed.

    """
    ...

def require_tokenizers(test_case):
    """
    Decorator marking a test that requires 🤗 Tokenizers.

    These tests are skipped when 🤗 Tokenizers isn't installed.

    """
    ...

def require_torch_multigpu(test_case):
    """
    Decorator marking a test that requires a multi-GPU setup (in PyTorch).

    These tests are skipped on a machine without multiple GPUs.

    To run *only* the multigpu tests, assuming all test names contain multigpu:
    $ pytest -sv ./tests -k "multigpu"
    """
    ...

def require_torch_non_multigpu(test_case):
    """
    Decorator marking a test that requires 0 or 1 GPU setup (in PyTorch).
    """
    ...

def require_torch_tpu(test_case):
    """
    Decorator marking a test that requires a TPU (in PyTorch).
    """
    ...

if _torch_available:
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"
else:
    torch_device = None
def require_torch_gpu(test_case):
    """Decorator marking a test that requires CUDA and PyTorch. """
    ...

def require_datasets(test_case):
    """Decorator marking a test that requires datasets."""
    ...

def require_faiss(test_case):
    """Decorator marking a test that requires faiss."""
    ...

def get_tests_dir(append_path=...):
    """
    Args:
        append_path: optional path to append to the tests dir path

    Return:
        The full path to the `tests` dir, so that the tests can be invoked from anywhere.
        Optionally `append_path` is joined after the `tests` dir the former is provided.

    """
    ...

def apply_print_resets(buf):
    ...

def assert_screenout(out, what):
    ...

class CaptureStd:
    """Context manager to capture:
    stdout, clean it up and make it available via obj.out
    stderr, and make it available via obj.err

    init arguments:
    - out - capture stdout: True/False, default True
    - err - capture stdout: True/False, default True

    Examples:

    with CaptureStdout() as cs:
        print("Secret message")
    print(f"captured: {cs.out}")

    import sys
    with CaptureStderr() as cs:
        print("Warning: ", file=sys.stderr)
    print(f"captured: {cs.err}")

    # to capture just one of the streams, but not the other
    with CaptureStd(err=False) as cs:
        print("Secret message")
    print(f"captured: {cs.out}")
    # but best use the stream-specific subclasses

    """
    def __init__(self, out=..., err=...) -> None:
        ...
    
    def __enter__(self):
        ...
    
    def __exit__(self, *exc):
        ...
    
    def __repr__(self):
        ...
    


class CaptureStdout(CaptureStd):
    """ Same as CaptureStd but captures only stdout """
    def __init__(self) -> None:
        ...
    


class CaptureStderr(CaptureStd):
    """ Same as CaptureStd but captures only stderr """
    def __init__(self) -> None:
        ...
    


class CaptureLogger:
    """Context manager to capture `logging` streams

    Args:
    - logger: 'logging` logger object

    Results:
        The captured output is available via `self.out`

    Example:

    >>> from transformers import logging
    >>> from transformers.testing_utils import CaptureLogger

    >>> msg = "Testing 1, 2, 3"
    >>> logging.set_verbosity_info()
    >>> logger = logging.get_logger("transformers.tokenization_bart")
    >>> with CaptureLogger(logger) as cl:
    ...     logger.info(msg)
    >>> assert cl.out, msg+"\n"
    """
    def __init__(self, logger) -> None:
        ...
    
    def __enter__(self):
        ...
    
    def __exit__(self, *exc):
        ...
    
    def __repr__(self):
        ...
    


class TestCasePlus(unittest.TestCase):
    """This class extends `unittest.TestCase` with additional features.

    Feature 1: Flexible auto-removable temp dirs which are guaranteed to get
    removed at the end of test.

    In all the following scenarios the temp dir will be auto-removed at the end
    of test, unless `after=False`.

    # 1. create a unique temp dir, `tmp_dir` will contain the path to the created temp dir
    def test_whatever(self):
        tmp_dir = self.get_auto_remove_tmp_dir()

    # 2. create a temp dir of my choice and delete it at the end - useful for debug when you want to
    # monitor a specific directory
    def test_whatever(self):
        tmp_dir = self.get_auto_remove_tmp_dir(tmp_dir="./tmp/run/test")

    # 3. create a temp dir of my choice and do not delete it at the end - useful for when you want
    # to look at the temp results
    def test_whatever(self):
        tmp_dir = self.get_auto_remove_tmp_dir(tmp_dir="./tmp/run/test", after=False)

    # 4. create a temp dir of my choice and ensure to delete it right away - useful for when you
    # disabled deletion in the previous test run and want to make sure the that tmp dir is empty
    # before the new test is run
    def test_whatever(self):
        tmp_dir = self.get_auto_remove_tmp_dir(tmp_dir="./tmp/run/test", before=True)

    Note 1: In order to run the equivalent of `rm -r` safely, only subdirs of the
    project repository checkout are allowed if an explicit `tmp_dir` is used, so
    that by mistake no `/tmp` or similar important part of the filesystem will
    get nuked. i.e. please always pass paths that start with `./`

    Note 2: Each test can register multiple temp dirs and they all will get
    auto-removed, unless requested otherwise.

    """
    def setUp(self):
        ...
    
    def get_auto_remove_tmp_dir(self, tmp_dir=..., after=..., before=...):
        """
        Args:
            tmp_dir (:obj:`string`, `optional`):
                use this path, if None a unique path will be assigned
            before (:obj:`bool`, `optional`, defaults to :obj:`False`):
                if `True` and tmp dir already exists make sure to empty it right away
            after (:obj:`bool`, `optional`, defaults to :obj:`True`):
                delete the tmp dir at the end of the test

        Returns:
            tmp_dir(:obj:`string`):
                either the same value as passed via `tmp_dir` or the path to the auto-created tmp dir
        """
        ...
    
    def tearDown(self):
        ...
    


def mockenv(**kwargs):
    """this is a convenience wrapper, that allows this:

    @mockenv(RUN_SLOW=True, USE_TF=False)
    def test_something():
        run_slow = os.getenv("RUN_SLOW", False)
        use_tf = os.getenv("USE_TF", False)
    """
    ...

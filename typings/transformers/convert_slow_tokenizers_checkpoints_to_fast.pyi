"""
This type stub file was generated by pyright.
"""

import argparse
import transformers
from transformers.convert_slow_tokenizer import SLOW_TO_FAST_CONVERTERS
from transformers.utils import logging

""" Convert slow tokenizers checkpoints in fast (serialization format of the `tokenizers` library) """
logger = logging.get_logger(__name__)
TOKENIZER_CLASSES = { name: getattr(transformers, name + "Fast") for name in SLOW_TO_FAST_CONVERTERS }
def convert_slow_checkpoint_to_fast(tokenizer_name, checkpoint_name, dump_path, force_download):
    ...

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

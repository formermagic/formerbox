"""
This type stub file was generated by pyright.
"""

import argparse
import fairseq
from packaging import version
from transformers.utils import logging

"""Convert RoBERTa checkpoint."""
if version.parse(fairseq.__version__) < version.parse("0.9.0"):
    ...
logger = logging.get_logger(__name__)
SAMPLE_TEXT = "Hello world! cécé herlolip"
def convert_roberta_checkpoint_to_pytorch(roberta_checkpoint_path: str, pytorch_dump_folder_path: str, classification_head: bool):
    """
    Copy/paste/tweak roberta's weights to our BERT structure.
    """
    ...

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

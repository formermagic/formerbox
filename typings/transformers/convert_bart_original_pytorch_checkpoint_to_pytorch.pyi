"""
This type stub file was generated by pyright.
"""

import argparse
import fairseq
import torch
from packaging import version
from transformers import BartForSequenceClassification, BartModel
from transformers.utils import logging

"""
This type stub file was generated by pyright.
"""
FAIRSEQ_MODELS = ["bart.large", "bart.large.mnli", "bart.large.cnn", "bart_xsum/model.pt"]
extra_arch = { "bart.large": BartModel,"bart.large.mnli": BartForSequenceClassification }
if version.parse(fairseq.__version__) < version.parse("0.9.0"):
    ...
logger = logging.get_logger(__name__)
SAMPLE_TEXT = " Hello world! cécé herlolip"
mnli_rename_keys = [("model.classification_heads.mnli.dense.weight", "classification_head.dense.weight"), ("model.classification_heads.mnli.dense.bias", "classification_head.dense.bias"), ("model.classification_heads.mnli.out_proj.weight", "classification_head.out_proj.weight"), ("model.classification_heads.mnli.out_proj.bias", "classification_head.out_proj.bias")]
def remove_ignore_keys_(state_dict):
    ...

def rename_key(dct, old, new):
    ...

def load_xsum_checkpoint(checkpoint_path):
    """Checkpoint path should end in model.pt"""
    ...

@torch.no_grad()
def convert_bart_checkpoint(checkpoint_path, pytorch_dump_folder_path, hf_checkpoint_name=...):
    """
    Copy/paste/tweak model's weights to our BERT structure.
    """
    ...

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

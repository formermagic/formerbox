"""
This type stub file was generated by pyright.
"""

import argparse
from transformers.utils import logging

"""
This type stub file was generated by pyright.
"""
logger = logging.get_logger(__name__)
def load_tf2_weights_in_bert(model, tf_checkpoint_path, config):
    ...

def convert_tf2_checkpoint_to_pytorch(tf_checkpoint_path, config_path, pytorch_dump_path):
    ...

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

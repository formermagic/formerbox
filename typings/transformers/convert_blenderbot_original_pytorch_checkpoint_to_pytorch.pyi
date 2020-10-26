"""
This type stub file was generated by pyright.
"""

import argparse
import torch
from transformers.utils import logging

"""Convert Blenderbot checkpoint."""
logger = logging.get_logger(__name__)
PATTERNS = [["attention", "attn"], ["encoder_attention", "encoder_attn"], ["q_lin", "q_proj"], ["k_lin", "k_proj"], ["v_lin", "v_proj"], ["out_lin", "out_proj"], ["norm_embeddings", "layernorm_embedding"], ["position_embeddings", "embed_positions"], ["embeddings", "embed_tokens"], ["ffn.lin", "fc"]]
def rename_state_dict_key(k):
    ...

def rename_layernorm_keys(sd):
    ...

IGNORE_KEYS = ["START"]
@torch.no_grad()
def convert_parlai_checkpoint(checkpoint_path, pytorch_dump_folder_path, config_json_path):
    """
    Copy/paste/tweak model's weights to our BERT structure.
    """
    ...

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
"""
This type stub file was generated by pyright.
"""

import argparse

"""Convert LXMERT checkpoint."""
def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, config_file, pytorch_dump_path):
    ...

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
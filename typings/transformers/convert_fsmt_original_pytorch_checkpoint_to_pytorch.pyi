"""
This type stub file was generated by pyright.
"""

import argparse

"""
This type stub file was generated by pyright.
"""
json_indent = 2
best_score_hparams = { "wmt19-ru-en": { "length_penalty": 1.1 },"wmt19-en-ru": { "length_penalty": 1.15 },"wmt19-en-de": { "length_penalty": 1 },"wmt19-de-en": { "length_penalty": 1.1 },"wmt16-en-de-dist-12-1": { "length_penalty": 0.6 },"wmt16-en-de-dist-6-1": { "length_penalty": 0.6 },"wmt16-en-de-12-1": { "length_penalty": 0.8 },"wmt19-de-en-6-6-base": { "length_penalty": 0.6 },"wmt19-de-en-6-6-big": { "length_penalty": 0.6 } }
org_names = {  }
def rewrite_dict_keys(d):
    ...

def convert_fsmt_checkpoint_to_pytorch(fsmt_checkpoint_path, pytorch_dump_folder_path):
    ...

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

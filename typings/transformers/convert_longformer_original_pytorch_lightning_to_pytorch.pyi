"""
This type stub file was generated by pyright.
"""

import argparse
import pytorch_lightning as pl

"""
This type stub file was generated by pyright.
"""
class LightningModel(pl.LightningModule):
    def __init__(self, model) -> None:
        ...
    
    def forward(self):
        ...
    


def convert_longformer_qa_checkpoint_to_pytorch(longformer_model: str, longformer_question_answering_ckpt_path: str, pytorch_dump_folder_path: str):
    ...

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

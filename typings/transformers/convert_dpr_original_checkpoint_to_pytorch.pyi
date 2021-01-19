"""
This type stub file was generated by pyright.
"""

import argparse
import collections
from pathlib import Path

"""
This type stub file was generated by pyright.
"""
CheckpointState = collections.namedtuple("CheckpointState", ["model_dict", "optimizer_dict", "scheduler_dict", "offset", "epoch", "encoder_params"])
def load_states_from_checkpoint(model_file: str) -> CheckpointState:
    ...

class DPRState:
    def __init__(self, src_file: Path) -> None:
        ...
    
    def load_dpr_model(self):
        ...
    
    @staticmethod
    def from_type(comp_type: str, *args, **kwargs) -> DPRState:
        ...
    


class DPRContextEncoderState(DPRState):
    def load_dpr_model(self):
        ...
    


class DPRQuestionEncoderState(DPRState):
    def load_dpr_model(self):
        ...
    


class DPRReaderState(DPRState):
    def load_dpr_model(self):
        ...
    


def convert(comp_type: str, src_file: Path, dest_dir: Path):
    ...

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    src_file = Path(args.src)
    dest_dir = <Expression>
    dest_dir = Path(dest_dir)

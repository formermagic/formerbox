"""
This type stub file was generated by pyright.
"""

from dataclasses import dataclass
from ..utils import logging

logger = logging.get_logger(__name__)
def list_field(default=..., metadata=...):
    ...

@dataclass
class BenchmarkArguments:
    """
    BenchMarkArguments are arguments we use in our benchmark scripts
    **which relate to the training loop itself**.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    def to_json_string(self):
        """
        Serializes this instance to a JSON string.
        """
        ...
    
    @property
    def model_names(self):
        ...
    
    @property
    def do_multi_processing(self):
        ...
    



"""
This type stub file was generated by pyright.
"""

from .file_utils import add_start_docstrings
from .modeling_bart import BartForConditionalGeneration

""""BlenderbotForConditionalGeneration which inherits from BART"""
BLENDER_START_DOCSTRING = r"""

    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)

    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__ subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general
    usage and behavior.

"""
BLENDERBOT_PRETRAINED_MODEL_ARCHIVE_LIST = ["facebook/blenderbot-3B", "facebook/blenderbot-90M"]
@add_start_docstrings("The BART Model with a language modeling head. Can be used for summarization.", BLENDER_START_DOCSTRING)
class BlenderbotForConditionalGeneration(BartForConditionalGeneration):
    """
    This class overrides :class:`~transformers.BartForConditionalGeneration`. Please check the
    superclass for the appropriate documentation alongside usage examples.
    """
    config_class = ...
    def adjust_logits_during_generation(self, logits, cur_len, max_length):
        ...
    


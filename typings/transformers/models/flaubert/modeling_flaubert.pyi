"""
This type stub file was generated by pyright.
"""

from ...file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from ...modeling_outputs import BaseModelOutput
from ...utils import logging
from ..xlm.modeling_xlm import XLMForMultipleChoice, XLMForQuestionAnswering, XLMForQuestionAnsweringSimple, XLMForSequenceClassification, XLMForTokenClassification, XLMModel, XLMWithLMHeadModel

""" PyTorch Flaubert model, based on XLM. """
logger = logging.get_logger(__name__)
_CONFIG_FOR_DOC = "FlaubertConfig"
_TOKENIZER_FOR_DOC = "FlaubertTokenizer"
FLAUBERT_PRETRAINED_MODEL_ARCHIVE_LIST = ["flaubert/flaubert_small_cased", "flaubert/flaubert_base_uncased", "flaubert/flaubert_base_cased", "flaubert/flaubert_large_cased"]
FLAUBERT_START_DOCSTRING = r"""

    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)

    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.

    Parameters:
        config (:class:`~transformers.FlaubertConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
"""
FLAUBERT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`~transformers.FlaubertTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in ``[0,
            1]``:

            - 0 corresponds to a `sentence A` token,
            - 1 corresponds to a `sentence B` token.

            `What are token type IDs? <../glossary.html#token-type-ids>`_
        position_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
            config.max_position_embeddings - 1]``.

            `What are position IDs? <../glossary.html#position-ids>`_
        lengths (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Length of each sentence that can be used to avoid performing attention on padding token indices. You can
            also use :obj:`attention_mask` for the same result (see above), kept here for compatibility. Indices
            selected in ``[0, ..., input_ids.size(-1)]``:
        cache (:obj:`Dict[str, torch.FloatTensor]`, `optional`):
            Dictionary strings to ``torch.FloatTensor`` that contains precomputed hidden-states (key and values in the
            attention blocks) as computed by the model (see :obj:`cache` output below). Can be used to speed up
            sequential decoding. The dictionary object will be modified in-place during the forward pass to add newly
            computed hidden-states.
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in ``[0, 1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
"""
@add_start_docstrings("The bare Flaubert Model transformer outputting raw hidden-states without any specific head on top.", FLAUBERT_START_DOCSTRING)
class FlaubertModel(XLMModel):
    config_class = ...
    def __init__(self, config) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(FLAUBERT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint="flaubert/flaubert_base_cased", output_type=BaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., langs=..., token_type_ids=..., position_ids=..., lengths=..., cache=..., head_mask=..., inputs_embeds=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        ...
    


@add_start_docstrings("""
    The Flaubert Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """, FLAUBERT_START_DOCSTRING)
class FlaubertWithLMHeadModel(XLMWithLMHeadModel):
    """
    This class overrides :class:`~transformers.XLMWithLMHeadModel`. Please check the superclass for the appropriate
    documentation alongside usage examples.
    """
    config_class = ...
    def __init__(self, config) -> None:
        ...
    


@add_start_docstrings("""
    Flaubert Model with a sequence classification/regression head on top (a linear layer on top of the pooled output)
    e.g. for GLUE tasks.
    """, FLAUBERT_START_DOCSTRING)
class FlaubertForSequenceClassification(XLMForSequenceClassification):
    """
    This class overrides :class:`~transformers.XLMForSequenceClassification`. Please check the superclass for the
    appropriate documentation alongside usage examples.
    """
    config_class = ...
    def __init__(self, config) -> None:
        ...
    


@add_start_docstrings("""
    Flaubert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """, FLAUBERT_START_DOCSTRING)
class FlaubertForTokenClassification(XLMForTokenClassification):
    """
    This class overrides :class:`~transformers.XLMForTokenClassification`. Please check the superclass for the
    appropriate documentation alongside usage examples.
    """
    config_class = ...
    def __init__(self, config) -> None:
        ...
    


@add_start_docstrings("""
    Flaubert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """, FLAUBERT_START_DOCSTRING)
class FlaubertForQuestionAnsweringSimple(XLMForQuestionAnsweringSimple):
    """
    This class overrides :class:`~transformers.XLMForQuestionAnsweringSimple`. Please check the superclass for the
    appropriate documentation alongside usage examples.
    """
    config_class = ...
    def __init__(self, config) -> None:
        ...
    


@add_start_docstrings("""
    Flaubert Model with a beam-search span classification head on top for extractive question-answering tasks like
    SQuAD (a linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """, FLAUBERT_START_DOCSTRING)
class FlaubertForQuestionAnswering(XLMForQuestionAnswering):
    """
    This class overrides :class:`~transformers.XLMForQuestionAnswering`. Please check the superclass for the
    appropriate documentation alongside usage examples.
    """
    config_class = ...
    def __init__(self, config) -> None:
        ...
    


@add_start_docstrings("""
    Flaubert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """, FLAUBERT_START_DOCSTRING)
class FlaubertForMultipleChoice(XLMForMultipleChoice):
    """
    This class overrides :class:`~transformers.XLMForMultipleChoice`. Please check the superclass for the appropriate
    documentation alongside usage examples.
    """
    config_class = ...
    def __init__(self, config) -> None:
        ...
    


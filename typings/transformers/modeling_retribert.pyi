"""
This type stub file was generated by pyright.
"""

from .file_utils import add_start_docstrings
from .modeling_utils import PreTrainedModel
from .utils import logging

"""
This type stub file was generated by pyright.
"""
logger = logging.get_logger(__name__)
RETRIBERT_PRETRAINED_MODEL_ARCHIVE_LIST = ["yjernite/retribert-base-uncased"]
class RetriBertPreTrainedModel(PreTrainedModel):
    """An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
    """
    config_class = ...
    load_tf_weights = ...
    base_model_prefix = ...


RETRIBERT_START_DOCSTRING = r"""

    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)

    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__ subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general
    usage and behavior.

    Parameters:
        config (:class:`~transformers.RetriBertConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""
@add_start_docstrings("""Bert Based model to embed queries or document for document retreival. """, RETRIBERT_START_DOCSTRING)
class RetriBertModel(RetriBertPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    def embed_sentences_checkpointed(self, input_ids, attention_mask, sent_encoder, checkpoint_batch_size=...):
        ...
    
    def embed_questions(self, input_ids, attention_mask=..., checkpoint_batch_size=...):
        ...
    
    def embed_answers(self, input_ids, attention_mask=..., checkpoint_batch_size=...):
        ...
    
    def forward(self, input_ids_query, attention_mask_query, input_ids_doc, attention_mask_doc, checkpoint_batch_size=...):
        r"""
        Args:
            input_ids_query (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary for the queries in a batch.

                Indices can be obtained using :class:`~transformers.RetriBertTokenizer`.
                See :meth:`transformers.PreTrainedTokenizer.encode` and
                :meth:`transformers.PreTrainedTokenizer.__call__` for details.

                `What are input IDs? <../glossary.html#input-ids>`__
            attention_mask_query (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices.
                Mask values selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            input_ids_doc (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary for the documents in a batch.
            attention_mask_doc (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on documents padding token indices.
            checkpoint_batch_size (:obj:`int`, `optional`, defaults to `:obj:`-1`):
                If greater than 0, uses gradient checkpointing to only compute sequence representation on
                :obj:`checkpoint_batch_size` examples at a time on the GPU. All query representations are still
                compared to all document representations in the batch.

        Return:
            :obj:`torch.FloatTensor`: The bidirectional cross-entropy loss obtained while trying to match each query to
            its corresponding document and each cocument to its corresponding query in the batch
        """
        ...
    



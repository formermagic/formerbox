"""
This type stub file was generated by pyright.
"""

from .configuration_utils import PretrainedConfig
from .file_utils import add_start_docstrings

""" RAG model configuration """
RAG_CONFIG_DOC = r"""
    :class:`~transformers.RagConfig` stores the configuration of a `RagModel`.
    Configuration objects inherit from  :class:`~transformers.PretrainedConfig` and can be used
    to control the model outputs. Read the documentation from  :class:`~transformers.PretrainedConfig`
    for more information.

    Args:
        title_sep (:obj:`str`, `optional`, defaults to  ``" / "``):
            Separator inserted between the title and the text of the retrieved document when calling :class:`~transformers.RagRetriever`.
        doc_sep (:obj:`str`, `optional`, defaults to  ``" // "``):
            Separator inserted between the the text of the retrieved document and the original input when calliang :class:`~transformers.RagRetriever`.
        n_docs (:obj:`int`, `optional`, defaults to 5):
            Number of documents to retrieve.
        max_combined_length (:obj:`int`, `optional`, defaults to 300):
            Max length of contextualized input returned by :meth:`~transformers.RagRetriever.__call__`.
        retrieval_vector_size (:obj:`int`, `optional`, defaults to 768):
            Dimensionality of the document embeddings indexed by :class:`~transformers.RagRetriever`.
        retrieval_batch_size (:obj:`int`, `optional`, defaults to 8):
            Retrieval batch size, defined as the number of queries issues concurrently to the faiss index excapsulated
            :class:`~transformers.RagRetriever`.
        dataset (:obj:`str`, `optional`, defaults to :obj:`"wiki_dpr"`):
            A dataset identifier of the indexed dataset in HuggingFace Datasets (list all available datasets and
            ids using :obj:`datasets.list_datasets()`).
        dataset_split (:obj:`str`, `optional`, defaults to :obj:`"train"`)
            Which split of the :obj:`dataset` to load.
        index_name (:obj:`str`, `optional`, defaults to :obj:`"compressed"`)
            The index name of the index associated with the :obj:`dataset`. One can choose between :obj:`"legacy"`,
            :obj:`"exact"` and :obj:`"compressed"`.
        index_path (:obj:`str`, `optional`)
            The path to the serialized faiss index on disk.
        passages_path: (:obj:`str`, `optional`):
            A path to text passages compatible with the faiss index. Required if using
            :class:`~transformers.retrieval_rag.LegacyIndex`
        use_dummy_dataset (:obj:`bool`, `optional`, defaults to ``False``)
            Whether to load a "dummy" variant of the dataset specified by :obj:`dataset`.
        label_smoothing (:obj:`float`, `optional`, defaults to 0.0):
            Only relevant if ``return_loss`` is set to :obj:`True`. Controls the ``epsilon`` parameter value for label
            smoothing in the loss calculation. If set to 0, no label smoothing is performed.
        do_marginalize (:obj:`bool`, `optional`, defaults to :obj:`False`):
            If :obj:`True`, the logits are marginalized over all documents
            by making use of ``torch.nn.functional.log_softmax``.
        reduce_loss (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to reduce the NLL loss using the ``torch.Tensor.sum`` operation.
        do_deduplication (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to deduplicate the generations from different context documents for a given input.
            Has to be set to :obj:`False` if used while training with distributed backend.
        exclude_bos_score (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to disregard the BOS token when computing the loss.
        output_retrieved(:obj:`bool`, `optional`, defaults to :obj:`False`):
            If set to ``True``, :obj:`retrieved_doc_embeds`, :obj:`retrieved_doc_ids`, :obj:`context_input_ids` and
            :obj:`context_attention_mask` are returned. See returned tensors for more detail.
"""
@add_start_docstrings(RAG_CONFIG_DOC)
class RagConfig(PretrainedConfig):
    model_type = ...
    def __init__(self, vocab_size=..., is_encoder_decoder=..., prefix=..., bos_token_id=..., pad_token_id=..., eos_token_id=..., decoder_start_token_id=..., title_sep=..., doc_sep=..., n_docs=..., max_combined_length=..., retrieval_vector_size=..., retrieval_batch_size=..., dataset=..., dataset_split=..., index_name=..., index_path=..., passages_path=..., use_dummy_dataset=..., reduce_loss=..., label_smoothing=..., do_deduplication=..., exclude_bos_score=..., do_marginalize=..., output_retrieved=..., **kwargs) -> None:
        ...
    
    @classmethod
    def from_question_encoder_generator_configs(cls, question_encoder_config: PretrainedConfig, generator_config: PretrainedConfig, **kwargs) -> PretrainedConfig:
        r"""
        Instantiate a :class:`~transformers.EncoderDecoderConfig` (or a derived class) from a pre-trained encoder model configuration and decoder model configuration.

        Returns:
            :class:`EncoderDecoderConfig`: An instance of a configuration object
        """
        ...
    
    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default :meth:`~transformers.PretrainedConfig.to_dict`.

        Returns:
            :obj:`Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        ...
    


"""
This type stub file was generated by pyright.
"""

import tensorflow as tf
from ...file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from ...modeling_tf_outputs import TFBaseModelOutputWithPast, TFCausalLMOutputWithPast, TFSequenceClassifierOutput
from ...modeling_tf_utils import TFCausalLanguageModelingLoss, TFPreTrainedModel, TFSequenceClassificationLoss, keras_serializable
from ...utils import logging

""" TF 2.0 CTRL model."""
logger = logging.get_logger(__name__)
_CONFIG_FOR_DOC = "CTRLConfig"
_TOKENIZER_FOR_DOC = "CTRLTokenizer"
TF_CTRL_PRETRAINED_MODEL_ARCHIVE_LIST = ["ctrl"]
def angle_defn(pos, i, d_model_size):
    ...

def positional_encoding(position, d_model_size):
    ...

def scaled_dot_product_attention(q, k, v, mask, attention_mask=..., head_mask=...):
    ...

class TFMultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model_size, num_heads, output_attentions=..., **kwargs) -> None:
        ...
    
    def split_into_heads(self, x, batch_size):
        ...
    
    def call(self, v, k, q, mask, layer_past, attention_mask, head_mask, use_cache, output_attentions, training=...):
        ...
    


class TFPointWiseFeedForwardLayer(tf.keras.layers.Layer):
    def __init__(self, d_model_size, dff, **kwargs) -> None:
        ...
    
    def call(self, inputs, trainable=...):
        ...
    


class TFEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model_size, num_heads, dff, rate=..., layer_norm_epsilon=..., output_attentions=..., **kwargs) -> None:
        ...
    
    def call(self, x, mask, layer_past, attention_mask, head_mask, use_cache, output_attentions, training=...):
        ...
    


@keras_serializable
class TFCTRLMainLayer(tf.keras.layers.Layer):
    config_class = ...
    def __init__(self, config, **kwargs) -> None:
        ...
    
    def get_input_embeddings(self):
        ...
    
    def set_input_embeddings(self, value):
        ...
    
    def call(self, input_ids=..., past=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., use_cache=..., output_attentions=..., output_hidden_states=..., return_dict=..., training=..., **kwargs):
        ...
    


class TFCTRLPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = ...
    base_model_prefix = ...


CTRL_START_DOCSTRING = r"""

    This model inherits from :class:`~transformers.TFPreTrainedModel`. Check the superclass documentation for the
    generic methods the library implements for all its model (such as downloading or saving, resizing the input
    embeddings, pruning heads etc.)

    This model is also a `tf.keras.Model <https://www.tensorflow.org/api_docs/python/tf/keras/Model>`__ subclass. Use
    it as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general usage
    and behavior.

    .. note::

        TF 2.0 models accepts two formats as inputs:

        - having all inputs as keyword arguments (like PyTorch models), or
        - having all inputs as a list, tuple or dict in the first positional arguments.

        This second option is useful when using :meth:`tf.keras.Model.fit` method which currently requires having all
        the tensors in the first argument of the model call function: :obj:`model(inputs)`.

        If you choose this second option, there are three possibilities you can use to gather all the input Tensors in
        the first positional argument :

        - a single Tensor with :obj:`input_ids` only and nothing else: :obj:`model(inputs_ids)`
        - a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:
          :obj:`model([input_ids, attention_mask])` or :obj:`model([input_ids, attention_mask, token_type_ids])`
        - a dictionary with one or several input Tensors associated to the input names given in the docstring:
          :obj:`model({"input_ids": input_ids, "token_type_ids": token_type_ids})`

    Parameters:
        config (:class:`~transformers.CTRLConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
"""
CTRL_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`Numpy array` or :obj:`tf.Tensor` of shape :obj:`(batch_size, input_ids_length)`):
            :obj:`input_ids_length` = ``sequence_length`` if ``past`` is ``None`` else ``past[0].shape[-2]``
            (``sequence_length`` of input past key value states).

            Indices of input sequence tokens in the vocabulary.

            If :obj:`past` is used, only input IDs that do not have their past calculated should be passed as
            ``input_ids``.

            Indices can be obtained using :class:`~transformers.CTRLTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.__call__` and :meth:`transformers.PreTrainedTokenizer.encode` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        past (:obj:`List[tf.Tensor]` of length :obj:`config.n_layers`):
            Contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model (see
            :obj:`past` output below). Can be used to speed up sequential decoding. The token ids which have their past
            given to this model should not be passed as input ids as they have already been computed.
        attention_mask (:obj:`tf.Tensor` or :obj:`Numpy array` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`tf.Tensor` or :obj:`Numpy array` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in ``[0,
            1]``:

            - 0 corresponds to a `sentence A` token,
            - 1 corresponds to a `sentence B` token.

            `What are token type IDs? <../glossary.html#token-type-ids>`__
        position_ids (:obj:`tf.Tensor` or :obj:`Numpy array` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
            config.max_position_embeddings - 1]``.

            `What are position IDs? <../glossary.html#position-ids>`__
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in ``[0, 1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (:obj:`tf.Tensor` or :obj:`Numpy array` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, ``past`` key value states are returned and can be used to speed up decoding (see
            ``past``).
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
        training (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to use the model in training mode (some modules like dropout modules have different
            behaviors between training and evaluation).
"""
@add_start_docstrings("The bare CTRL Model transformer outputting raw hidden-states without any specific head on top.", CTRL_START_DOCSTRING)
class TFCTRLModel(TFCTRLPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(CTRL_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint="ctrl", output_type=TFBaseModelOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids=..., past=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., use_cache=..., output_attentions=..., output_hidden_states=..., return_dict=..., training=..., **kwargs):
        ...
    
    def serving_output(self, output):
        ...
    


class TFCTRLLMHead(tf.keras.layers.Layer):
    def __init__(self, config, input_embeddings, **kwargs) -> None:
        ...
    
    def build(self, input_shape):
        ...
    
    def get_output_embeddings(self):
        ...
    
    def set_output_embeddings(self, value):
        ...
    
    def get_bias(self):
        ...
    
    def set_bias(self, value):
        ...
    
    def call(self, hidden_states):
        ...
    


@add_start_docstrings("""
    The CTRL Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """, CTRL_START_DOCSTRING)
class TFCTRLLMHeadModel(TFCTRLPreTrainedModel, TFCausalLanguageModelingLoss):
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...
    
    def get_lm_head(self):
        ...
    
    def get_prefix_bias_name(self):
        ...
    
    def prepare_inputs_for_generation(self, inputs, past, **kwargs):
        ...
    
    @add_start_docstrings_to_model_forward(CTRL_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint="ctrl", output_type=TFCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids=..., past=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., use_cache=..., output_attentions=..., output_hidden_states=..., return_dict=..., labels=..., training=..., **kwargs):
        r"""
        labels (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the cross entropy classification loss. Indices should be in ``[0, ...,
            config.vocab_size - 1]``.
        """
        ...
    
    def serving_output(self, output):
        ...
    


@add_start_docstrings("""
    The CTRL Model transformer with a sequence classification head on top (linear layer).

    :class:`~transformers.TFCTRLForSequenceClassification` uses the last token in order to do the classification, as
    other causal models (e.g. GPT-1, GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    :obj:`pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each
    row. If no :obj:`pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot
    guess the padding tokens when :obj:`inputs_embeds` are passed instead of :obj:`input_ids`, it does the same (take
    the last value in each row of the batch).
    """, CTRL_START_DOCSTRING)
class TFCTRLForSequenceClassification(TFCTRLPreTrainedModel, TFSequenceClassificationLoss):
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...
    
    def get_output_embeddings(self):
        ...
    
    @add_start_docstrings_to_model_forward(CTRL_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint="ctrl", output_type=TFSequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids=..., past=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., use_cache=..., output_attentions=..., output_hidden_states=..., return_dict=..., labels=..., training=..., **kwargs):
        r"""
        labels (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the cross entropy classification loss. Indices should be in ``[0, ...,
            config.vocab_size - 1]``.
        """
        ...
    
    def serving_output(self, output):
        ...
    


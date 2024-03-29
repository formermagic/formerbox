"""
This type stub file was generated by pyright.
"""

import tensorflow as tf
from dataclasses import dataclass
from typing import Optional, Tuple
from ...file_utils import ModelOutput, add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from ...modeling_tf_outputs import TFBaseModelOutput, TFMaskedLMOutput, TFMultipleChoiceModelOutput, TFQuestionAnsweringModelOutput, TFSequenceClassifierOutput, TFTokenClassifierOutput
from ...modeling_tf_utils import TFMaskedLanguageModelingLoss, TFMultipleChoiceLoss, TFPreTrainedModel, TFQuestionAnsweringLoss, TFSequenceClassificationLoss, TFTokenClassificationLoss, keras_serializable
from ...utils import logging

""" TF 2.0 Funnel model. """
logger = logging.get_logger(__name__)
_CONFIG_FOR_DOC = "FunnelConfig"
_TOKENIZER_FOR_DOC = "FunnelTokenizer"
TF_FUNNEL_PRETRAINED_MODEL_ARCHIVE_LIST = ["funnel-transformer/small", "funnel-transformer/small-base", "funnel-transformer/medium", "funnel-transformer/medium-base", "funnel-transformer/intermediate", "funnel-transformer/intermediate-base", "funnel-transformer/large", "funnel-transformer/large-base", "funnel-transformer/xlarge-base", "funnel-transformer/xlarge"]
INF = 1000000
class TFFunnelEmbeddings(tf.keras.layers.Layer):
    """Construct the embeddings from word embeddings."""
    def __init__(self, config, **kwargs) -> None:
        ...
    
    def build(self, input_shape):
        """Build shared word embedding layer """
        ...
    
    def call(self, input_ids=..., inputs_embeds=..., mode=..., training=...):
        """
        Get token embeddings of inputs

        Args:
            inputs: list of three int64 tensors with shape [batch_size, length]: (input_ids, position_ids, token_type_ids)
            mode: string, a valid value is one of "embedding" and "linear"

        Returns:
            outputs: (1) If mode == "embedding", output embedding tensor, float32 with shape [batch_size, length,
            embedding_size]; (2) mode == "linear", output linear tensor, float32 with shape [batch_size, length,
            vocab_size]

        Raises:
            ValueError: if mode is not valid.

        Shared weights logic adapted from
        https://github.com/tensorflow/models/blob/a009f4fb9d2fc4949e32192a944688925ef78659/official/transformer/v2/embedding_layer.py#L24
        """
        ...
    


class TFFunnelAttentionStructure:
    """
    Contains helpers for `TFFunnelRelMultiheadAttention `.
    """
    cls_token_type_id: int = ...
    def __init__(self, config) -> None:
        ...
    
    def init_attention_inputs(self, inputs_embeds, attention_mask=..., token_type_ids=..., training=...):
        """ Returns the attention inputs associated to the inputs of the model. """
        ...
    
    def token_type_ids_to_mat(self, token_type_ids):
        """Convert `token_type_ids` to `token_type_mat`."""
        ...
    
    def get_position_embeds(self, seq_len, dtype=..., training=...):
        """
        Create and cache inputs related to relative position encoding. Those are very different depending on whether we
        are using the factorized or the relative shift attention:

        For the factorized attention, it returns the matrices (phi, pi, psi, omega) used in the paper, appendix A.2.2,
        final formula.

        For the relative shif attention, it returns all possible vectors R used in the paper, appendix A.2.1, final
        formula.

        Paper link: https://arxiv.org/abs/2006.03236
        """
        ...
    
    def stride_pool_pos(self, pos_id, block_index):
        """
        Pool `pos_id` while keeping the cls token separate (if `self.separate_cls=True`).
        """
        ...
    
    def relative_pos(self, pos, stride, pooled_pos=..., shift=...):
        """
        Build the relative positional vector between `pos` and `pooled_pos`.
        """
        ...
    
    def stride_pool(self, tensor, axis):
        """
        Perform pooling by stride slicing the tensor along the given axis.
        """
        ...
    
    def pool_tensor(self, tensor, mode=..., stride=...):
        """Apply 1D pooling to a tensor of size [B x T (x H)]."""
        ...
    
    def pre_attention_pooling(self, output, attention_inputs):
        """ Pool `output` and the proper parts of `attention_inputs` before the attention layer. """
        ...
    
    def post_attention_pooling(self, attention_inputs):
        """ Pool the proper parts of `attention_inputs` after the attention layer. """
        ...
    


class TFFunnelRelMultiheadAttention(tf.keras.layers.Layer):
    def __init__(self, config, block_index, **kwargs) -> None:
        ...
    
    def build(self, input_shape):
        ...
    
    def relative_positional_attention(self, position_embeds, q_head, context_len, cls_mask=...):
        """ Relative attention score for the positional encodings """
        ...
    
    def relative_token_type_attention(self, token_type_mat, q_head, cls_mask=...):
        """ Relative attention score for the token_type_ids """
        ...
    
    def call(self, query, key, value, attention_inputs, output_attentions=..., training=...):
        ...
    


class TFFunnelPositionwiseFFN(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None:
        ...
    
    def call(self, hidden, training=...):
        ...
    


class TFFunnelLayer(tf.keras.layers.Layer):
    def __init__(self, config, block_index, **kwargs) -> None:
        ...
    
    def call(self, query, key, value, attention_inputs, output_attentions=..., training=...):
        ...
    


class TFFunnelEncoder(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None:
        ...
    
    def call(self, inputs_embeds, attention_mask=..., token_type_ids=..., output_attentions=..., output_hidden_states=..., return_dict=..., training=...):
        ...
    


def upsample(x, stride, target_len, separate_cls=..., truncate_seq=...):
    """
    Upsample tensor `x` to match `target_len` by repeating the tokens `stride` time on the sequence length dimension.
    """
    ...

class TFFunnelDecoder(tf.keras.layers.Layer):
    def __init__(self, config, **kwargs) -> None:
        ...
    
    def call(self, final_hidden, first_block_hidden, attention_mask=..., token_type_ids=..., output_attentions=..., output_hidden_states=..., return_dict=..., training=...):
        ...
    


@keras_serializable
class TFFunnelBaseLayer(tf.keras.layers.Layer):
    """ Base model without decoder """
    config_class = ...
    def __init__(self, config, **kwargs) -> None:
        ...
    
    def get_input_embeddings(self):
        ...
    
    def set_input_embeddings(self, value):
        ...
    
    def call(self, input_ids=..., attention_mask=..., token_type_ids=..., inputs_embeds=..., output_attentions=..., output_hidden_states=..., return_dict=..., training=..., **kwargs):
        ...
    


@keras_serializable
class TFFunnelMainLayer(tf.keras.layers.Layer):
    """ Base model with decoder """
    config_class = ...
    def __init__(self, config, **kwargs) -> None:
        ...
    
    def get_input_embeddings(self):
        ...
    
    def set_input_embeddings(self, value):
        ...
    
    def call(self, input_ids=..., attention_mask=..., token_type_ids=..., inputs_embeds=..., output_attentions=..., output_hidden_states=..., return_dict=..., training=..., **kwargs):
        ...
    


class TFFunnelDiscriminatorPredictions(tf.keras.layers.Layer):
    """Prediction module for the discriminator, made up of two dense layers."""
    def __init__(self, config, **kwargs) -> None:
        ...
    
    def call(self, discriminator_hidden_states):
        ...
    


class TFFunnelMaskedLMHead(tf.keras.layers.Layer):
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
    
    def call(self, hidden_states, training=...):
        ...
    


class TFFunnelClassificationHead(tf.keras.layers.Layer):
    def __init__(self, config, n_labels, **kwargs) -> None:
        ...
    
    def call(self, hidden, training=...):
        ...
    


class TFFunnelPreTrainedModel(TFPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = ...
    base_model_prefix = ...


@dataclass
class TFFunnelForPreTrainingOutput(ModelOutput):
    """
    Output type of :class:`~transformers.FunnelForPreTraining`.

    Args:
        logits (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length)`):
            Prediction scores of the head (scores for each token before SoftMax).
        hidden_states (:obj:`tuple(tf.ensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`tf.Tensor` (one for the output of the embeddings + one for the output of each layer) of
            shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`tf.Tensor` (one for each layer) of shape :obj:`(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    logits: tf.Tensor = ...
    hidden_states: Optional[Tuple[tf.Tensor]] = ...
    attentions: Optional[Tuple[tf.Tensor]] = ...


FUNNEL_START_DOCSTRING = r"""

    The Funnel Transformer model was proposed in `Funnel-Transformer: Filtering out Sequential Redundancy for Efficient
    Language Processing <https://arxiv.org/abs/2006.03236>`__ by Zihang Dai, Guokun Lai, Yiming Yang, Quoc V. Le.

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
        config (:class:`~transformers.XxxConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
"""
FUNNEL_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`Numpy array` or :obj:`tf.Tensor` of shape :obj:`({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`~transformers.FunnelTokenizer`. See
            :func:`transformers.PreTrainedTokenizer.__call__` and :func:`transformers.PreTrainedTokenizer.encode` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`Numpy array` or :obj:`tf.Tensor` of shape :obj:`({0})`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`Numpy array` or :obj:`tf.Tensor` of shape :obj:`({0})`, `optional`):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in ``[0,
            1]``:

            - 0 corresponds to a `sentence A` token,
            - 1 corresponds to a `sentence B` token.

            `What are token type IDs? <../glossary.html#token-type-ids>`__
        inputs_embeds (:obj:`tf.Tensor` of shape :obj:`({0}, hidden_size)`, `optional`):
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
        training (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to use the model in training mode (some modules like dropout modules have different
            behaviors between training and evaluation).
"""
@add_start_docstrings("""
    The base Funnel Transformer Model transformer outputting raw hidden-states without upsampling head (also called
    decoder) or any task-specific head on top.
    """, FUNNEL_START_DOCSTRING)
class TFFunnelBaseModel(TFFunnelPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(FUNNEL_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint="funnel-transformer/small-base", output_type=TFBaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids=..., attention_mask=..., token_type_ids=..., inputs_embeds=..., output_attentions=..., output_hidden_states=..., return_dict=..., training=..., **kwargs):
        ...
    
    def serving_output(self, output):
        ...
    


@add_start_docstrings("The bare Funnel Transformer Model transformer outputting raw hidden-states without any specific head on top.", FUNNEL_START_DOCSTRING)
class TFFunnelModel(TFFunnelPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(FUNNEL_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint="funnel-transformer/small", output_type=TFBaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids=..., attention_mask=..., token_type_ids=..., inputs_embeds=..., output_attentions=..., output_hidden_states=..., return_dict=..., training=..., **kwargs):
        ...
    
    def serving_output(self, output):
        ...
    


@add_start_docstrings("""
    Funnel model with a binary classification head on top as used during pretraining for identifying generated tokens.
    """, FUNNEL_START_DOCSTRING)
class TFFunnelForPreTraining(TFFunnelPreTrainedModel):
    def __init__(self, config, **kwargs) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(FUNNEL_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=TFFunnelForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids=..., attention_mask=..., token_type_ids=..., inputs_embeds=..., output_attentions=..., output_hidden_states=..., return_dict=..., training=..., **kwargs):
        r"""
        Returns:

        Examples::

            >>> from transformers import FunnelTokenizer, TFFunnelForPreTraining
            >>> import torch

            >>> tokenizer = TFFunnelTokenizer.from_pretrained('funnel-transformer/small')
            >>> model = TFFunnelForPreTraining.from_pretrained('funnel-transformer/small')

            >>> inputs = tokenizer("Hello, my dog is cute", return_tensors= "tf")
            >>> logits = model(inputs).logits
        """
        ...
    
    def serving_output(self, output):
        ...
    


@add_start_docstrings("""Funnel Model with a `language modeling` head on top. """, FUNNEL_START_DOCSTRING)
class TFFunnelForMaskedLM(TFFunnelPreTrainedModel, TFMaskedLanguageModelingLoss):
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...
    
    def get_lm_head(self):
        ...
    
    def get_prefix_bias_name(self):
        ...
    
    @add_start_docstrings_to_model_forward(FUNNEL_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint="funnel-transformer/small", output_type=TFMaskedLMOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids=..., attention_mask=..., token_type_ids=..., inputs_embeds=..., output_attentions=..., output_hidden_states=..., return_dict=..., labels=..., training=..., **kwargs):
        r"""
        labels (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        """
        ...
    
    def serving_output(self, output):
        ...
    


@add_start_docstrings("""
    Funnel Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """, FUNNEL_START_DOCSTRING)
class TFFunnelForSequenceClassification(TFFunnelPreTrainedModel, TFSequenceClassificationLoss):
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(FUNNEL_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint="funnel-transformer/small-base", output_type=TFSequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids=..., attention_mask=..., token_type_ids=..., inputs_embeds=..., output_attentions=..., output_hidden_states=..., return_dict=..., labels=..., training=..., **kwargs):
        r"""
        labels (:obj:`tf.Tensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        ...
    
    def serving_output(self, output):
        ...
    


@add_start_docstrings("""
    Funnel Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """, FUNNEL_START_DOCSTRING)
class TFFunnelForMultipleChoice(TFFunnelPreTrainedModel, TFMultipleChoiceLoss):
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...
    
    @property
    def dummy_inputs(self):
        """
        Dummy inputs to build the network.

        Returns:
            tf.Tensor with dummy inputs
        """
        ...
    
    @add_start_docstrings_to_model_forward(FUNNEL_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint="funnel-transformer/small-base", output_type=TFMultipleChoiceModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids=..., attention_mask=..., token_type_ids=..., inputs_embeds=..., output_attentions=..., output_hidden_states=..., return_dict=..., labels=..., training=..., **kwargs):
        r"""
        labels (:obj:`tf.Tensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices]`` where :obj:`num_choices` is the size of the second dimension of the input tensors. (See
            :obj:`input_ids` above)
        """
        ...
    
    @tf.function(input_signature=[{ "input_ids": tf.TensorSpec((None, None, None), tf.int32, name="input_ids"),"attention_mask": tf.TensorSpec((None, None, None), tf.int32, name="attention_mask"),"token_type_ids": tf.TensorSpec((None, None, None), tf.int32, name="token_type_ids") }])
    def serving(self, inputs):
        ...
    
    def serving_output(self, output):
        ...
    


@add_start_docstrings("""
    Funnel Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """, FUNNEL_START_DOCSTRING)
class TFFunnelForTokenClassification(TFFunnelPreTrainedModel, TFTokenClassificationLoss):
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(FUNNEL_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint="funnel-transformer/small", output_type=TFTokenClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids=..., attention_mask=..., token_type_ids=..., inputs_embeds=..., output_attentions=..., output_hidden_states=..., return_dict=..., labels=..., training=..., **kwargs):
        r"""
        labels (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        ...
    
    def serving_output(self, output):
        ...
    


@add_start_docstrings("""
    Funnel Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """, FUNNEL_START_DOCSTRING)
class TFFunnelForQuestionAnswering(TFFunnelPreTrainedModel, TFQuestionAnsweringLoss):
    def __init__(self, config, *inputs, **kwargs) -> None:
        ...
    
    @add_start_docstrings_to_model_forward(FUNNEL_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint="funnel-transformer/small", output_type=TFQuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, input_ids=..., attention_mask=..., token_type_ids=..., inputs_embeds=..., output_attentions=..., output_hidden_states=..., return_dict=..., start_positions=..., end_positions=..., training=..., **kwargs):
        r"""
        start_positions (:obj:`tf.Tensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        end_positions (:obj:`tf.Tensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        """
        ...
    
    def serving_output(self, output):
        ...
    



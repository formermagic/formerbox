"""
This type stub file was generated by pyright.
"""

from dataclasses import dataclass
from torch import nn
from .file_utils import ModelOutput, add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_callable, replace_return_docstrings
from .modeling_outputs import BaseModelOutputWithPooling, CausalLMOutput, MaskedLMOutput, MultipleChoiceModelOutput, NextSentencePredictorOutput, QuestionAnsweringModelOutput, SequenceClassifierOutput, TokenClassifierOutput
from .modeling_utils import PreTrainedModel
from .utils import logging

"""
This type stub file was generated by pyright.
"""
logger = logging.get_logger(__name__)
_CONFIG_FOR_DOC = "BertConfig"
_TOKENIZER_FOR_DOC = "BertTokenizer"
BERT_PRETRAINED_MODEL_ARCHIVE_LIST = ["bert-base-uncased", "bert-large-uncased", "bert-base-cased", "bert-large-cased", "bert-base-multilingual-uncased", "bert-base-multilingual-cased", "bert-base-chinese", "bert-base-german-cased", "bert-large-uncased-whole-word-masking", "bert-large-cased-whole-word-masking", "bert-large-uncased-whole-word-masking-finetuned-squad", "bert-large-cased-whole-word-masking-finetuned-squad", "bert-base-cased-finetuned-mrpc", "bert-base-german-dbmdz-cased", "bert-base-german-dbmdz-uncased", "cl-tohoku/bert-base-japanese", "cl-tohoku/bert-base-japanese-whole-word-masking", "cl-tohoku/bert-base-japanese-char", "cl-tohoku/bert-base-japanese-char-whole-word-masking", "TurkuNLP/bert-base-finnish-cased-v1", "TurkuNLP/bert-base-finnish-uncased-v1", "wietsedv/bert-base-dutch-cased"]
def load_tf_weights_in_bert(model, config, tf_checkpoint_path):
    """Load tf checkpoints in a pytorch model."""
    ...

class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""
    def __init__(self, config) -> None:
        ...
    
    def forward(self, input_ids=..., token_type_ids=..., position_ids=..., inputs_embeds=...):
        ...
    


class BertSelfAttention(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def transpose_for_scores(self, x):
        ...
    
    def forward(self, hidden_states, attention_mask=..., head_mask=..., encoder_hidden_states=..., encoder_attention_mask=..., output_attentions=...):
        ...
    


class BertSelfOutput(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states, input_tensor):
        ...
    


class BertAttention(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def prune_heads(self, heads):
        ...
    
    def forward(self, hidden_states, attention_mask=..., head_mask=..., encoder_hidden_states=..., encoder_attention_mask=..., output_attentions=...):
        ...
    


class BertIntermediate(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states):
        ...
    


class BertOutput(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states, input_tensor):
        ...
    


class BertLayer(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states, attention_mask=..., head_mask=..., encoder_hidden_states=..., encoder_attention_mask=..., output_attentions=...):
        ...
    
    def feed_forward_chunk(self, attention_output):
        ...
    


class BertEncoder(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states, attention_mask=..., head_mask=..., encoder_hidden_states=..., encoder_attention_mask=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        ...
    


class BertPooler(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states):
        ...
    


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states):
        ...
    


class BertLMPredictionHead(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, hidden_states):
        ...
    


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, sequence_output):
        ...
    


class BertOnlyNSPHead(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, pooled_output):
        ...
    


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config) -> None:
        ...
    
    def forward(self, sequence_output, pooled_output):
        ...
    


class BertPreTrainedModel(PreTrainedModel):
    """An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
    """
    config_class = ...
    load_tf_weights = ...
    base_model_prefix = ...
    authorized_missing_keys = ...


@dataclass
class BertForPreTrainingOutput(ModelOutput):
    """
    Output type of :class:`~transformers.BertForPreTrainingModel`.

    Args:
        loss (`optional`, returned when ``labels`` is provided, ``torch.FloatTensor`` of shape :obj:`(1,)`):
            Total loss as the sum of the masked language modeling loss and the next sequence prediction
            (classification) loss.
        prediction_logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        seq_relationship_logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, 2)`):
            Prediction scores of the next sequence prediction (classification) head (scores of True/False
            continuation before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    ...


BERT_START_DOCSTRING = r"""

    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)

    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__ subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general
    usage and behavior.

    Parameters:
        config (:class:`~transformers.BertConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""
BERT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`~transformers.BertTokenizer`.
            See :meth:`transformers.PreTrainedTokenizer.encode` and
            :meth:`transformers.PreTrainedTokenizer.__call__` for details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`({0})`, `optional`):
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`, `optional`):
            Segment token indices to indicate first and second portions of the inputs.
            Indices are selected in ``[0, 1]``:

            - 0 corresponds to a `sentence A` token,
            - 1 corresponds to a `sentence B` token.

            `What are token type IDs? <../glossary.html#token-type-ids>`_
        position_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`, `optional`):
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range ``[0, config.max_position_embeddings - 1]``.

            `What are position IDs? <../glossary.html#position-ids>`_
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`({0}, hidden_size)`, `optional`):
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
@add_start_docstrings("The bare Bert Model transformer outputting raw hidden-states without any specific head on top.", BERT_START_DOCSTRING)
class BertModel(BertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well
    as a decoder, in which case a layer of cross-attention is added between
    the self-attention layers, following the architecture described in `Attention is all you need
    <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
    Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the
    :obj:`is_decoder` argument of the configuration set to :obj:`True`.
    To be used in a Seq2Seq model, the model needs to initialized with both :obj:`is_decoder`
    argument and :obj:`add_cross_attention` set to :obj:`True`; an
    :obj:`encoder_hidden_states` is then expected as an input to the forward pass.
    """
    def __init__(self, config, add_pooling_layer=...) -> None:
        ...
    
    def get_input_embeddings(self):
        ...
    
    def set_input_embeddings(self, value):
        ...
    
    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint="bert-base-uncased", output_type=BaseModelOutputWithPooling, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., encoder_hidden_states=..., encoder_attention_mask=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
            if the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask
            is used in the cross-attention if the model is configured as a decoder.
            Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        """
        ...
    


@add_start_docstrings("""Bert Model with two heads on top as done during the pre-training: a `masked language modeling` head and
    a `next sentence prediction (classification)` head. """, BERT_START_DOCSTRING)
class BertForPreTraining(BertPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    def get_output_embeddings(self):
        ...
    
    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=BertForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., labels=..., next_sentence_label=..., output_attentions=..., output_hidden_states=..., return_dict=..., **kwargs):
        r"""
        labels (:obj:`torch.LongTensor` of shape ``(batch_size, sequence_length)``, `optional`):
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``
        next_sentence_label (``torch.LongTensor`` of shape ``(batch_size,)``, `optional`):
            Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair (see :obj:`input_ids` docstring)
            Indices should be in ``[0, 1]``:

            - 0 indicates sequence B is a continuation of sequence A,
            - 1 indicates sequence B is a random sequence.
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Used to hide legacy arguments that have been deprecated.

        Returns:

        Example::

            >>> from transformers import BertTokenizer, BertForPreTraining
            >>> import torch

            >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            >>> model = BertForPreTraining.from_pretrained('bert-base-uncased', return_dict=True)

            >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
            >>> outputs = model(**inputs)

            >>> prediction_logits = outputs.prediction_logits
            >>> seq_relationship_logits = outputs.seq_relationship_logits
        """
        ...
    


@add_start_docstrings("""Bert Model with a `language modeling` head on top for CLM fine-tuning. """, BERT_START_DOCSTRING)
class BertLMHeadModel(BertPreTrainedModel):
    authorized_unexpected_keys = ...
    authorized_missing_keys = ...
    def __init__(self, config) -> None:
        ...
    
    def get_output_embeddings(self):
        ...
    
    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=CausalLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., encoder_hidden_states=..., encoder_attention_mask=..., labels=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
            if the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask
            is used in the cross-attention if the model is configured as a decoder.
            Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the left-to-right language modeling loss (next word prediction).
            Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
            n ``[0, ..., config.vocab_size]``

        Returns:

        Example::

            >>> from transformers import BertTokenizer, BertLMHeadModel, BertConfig
            >>> import torch

            >>> tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
            >>> config = BertConfig.from_pretrained("bert-base-cased")
            >>> config.is_decoder = True
            >>> model = BertLMHeadModel.from_pretrained('bert-base-cased', config=config, return_dict=True)

            >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
            >>> outputs = model(**inputs)

            >>> prediction_logits = outputs.logits
        """
        ...
    
    def prepare_inputs_for_generation(self, input_ids, attention_mask=..., **model_kwargs):
        ...
    


@add_start_docstrings("""Bert Model with a `language modeling` head on top. """, BERT_START_DOCSTRING)
class BertForMaskedLM(BertPreTrainedModel):
    authorized_unexpected_keys = ...
    authorized_missing_keys = ...
    def __init__(self, config) -> None:
        ...
    
    def get_output_embeddings(self):
        ...
    
    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint="bert-base-uncased", output_type=MaskedLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., encoder_hidden_states=..., encoder_attention_mask=..., labels=..., output_attentions=..., output_hidden_states=..., return_dict=..., **kwargs):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Used to hide legacy arguments that have been deprecated.
        """
        ...
    
    def prepare_inputs_for_generation(self, input_ids, attention_mask=..., **model_kwargs):
        ...
    


@add_start_docstrings("""Bert Model with a `next sentence prediction (classification)` head on top. """, BERT_START_DOCSTRING)
class BertForNextSentencePrediction(BertPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=NextSentencePredictorOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., next_sentence_label=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        r"""
        next_sentence_label (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair
            (see ``input_ids`` docstring).  Indices should be in ``[0, 1]``:

            - 0 indicates sequence B is a continuation of sequence A,
            - 1 indicates sequence B is a random sequence.

        Returns:

        Example::

            >>> from transformers import BertTokenizer, BertForNextSentencePrediction
            >>> import torch

            >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            >>> model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased', return_dict=True)

            >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
            >>> next_sentence = "The sky is blue due to the shorter wavelength of blue light."
            >>> encoding = tokenizer(prompt, next_sentence, return_tensors='pt')

            >>> outputs = model(**encoding, next_sentence_label=torch.LongTensor([1]))
            >>> logits = outputs.logits
            >>> assert logits[0, 0] < logits[0, 1] # next sentence was random
        """
        ...
    


@add_start_docstrings("""Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of
    the pooled output) e.g. for GLUE tasks. """, BERT_START_DOCSTRING)
class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint="bert-base-uncased", output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., labels=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        ...
    


@add_start_docstrings("""Bert Model with a multiple choice classification head on top (a linear layer on top of
    the pooled output and a softmax) e.g. for RocStories/SWAG tasks. """, BERT_START_DOCSTRING)
class BertForMultipleChoice(BertPreTrainedModel):
    def __init__(self, config) -> None:
        ...
    
    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint="bert-base-uncased", output_type=MultipleChoiceModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., labels=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss.
            Indices should be in ``[0, ..., num_choices-1]`` where :obj:`num_choices` is the size of the second dimension
            of the input tensors. (See :obj:`input_ids` above)
        """
        ...
    


@add_start_docstrings("""Bert Model with a token classification head on top (a linear layer on top of
    the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks. """, BERT_START_DOCSTRING)
class BertForTokenClassification(BertPreTrainedModel):
    authorized_unexpected_keys = ...
    def __init__(self, config) -> None:
        ...
    
    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint="bert-base-uncased", output_type=TokenClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., labels=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
        """
        ...
    


@add_start_docstrings("""Bert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear
    layers on top of the hidden-states output to compute `span start logits` and `span end logits`). """, BERT_START_DOCSTRING)
class BertForQuestionAnswering(BertPreTrainedModel):
    authorized_unexpected_keys = ...
    def __init__(self, config) -> None:
        ...
    
    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint="bert-base-uncased", output_type=QuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=..., attention_mask=..., token_type_ids=..., position_ids=..., head_mask=..., inputs_embeds=..., start_positions=..., end_positions=..., output_attentions=..., output_hidden_states=..., return_dict=...):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        """
        ...
    



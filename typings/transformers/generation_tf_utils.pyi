"""
This type stub file was generated by pyright.
"""

from .utils import logging

logger = logging.get_logger(__name__)
class TFGenerationMixin:
    """
    A class containing all of the functions supporting generation, to be used as a mixin in
    :class:`~transformers.TFPreTrainedModel`.
    """
    def prepare_inputs_for_generation(self, inputs, **kwargs):
        """
        Implement in subclasses of :class:`~transformers.TFPreTrainedModel` for custom behavior to prepare inputs in
        the generate method.
        """
        ...
    
    def generate(self, input_ids=..., max_length=..., min_length=..., do_sample=..., early_stopping=..., num_beams=..., temperature=..., top_k=..., top_p=..., repetition_penalty=..., bad_words_ids=..., bos_token_id=..., pad_token_id=..., eos_token_id=..., length_penalty=..., no_repeat_ngram_size=..., num_return_sequences=..., attention_mask=..., decoder_start_token_id=..., use_cache=...):
        r"""
        Generates sequences for models with a language modeling head. The method currently supports greedy decoding,
        beam-search decoding, sampling with temperature, sampling with top-k or nucleus sampling.

        Adapted in part from `Facebook's XLM beam search code
        <https://github.com/facebookresearch/XLM/blob/9e6f6814d17be4fe5b15f2e6c43eb2b2d76daeb4/src/model/transformer.py#L529>`__.

        Apart from :obj:`input_ids` and :obj:`attention_mask`, all the arguments below will default to the value of the
        attribute of the same name inside the :class:`~transformers.PretrainedConfig` of the model. The default values
        indicated are the default values of those config.

        Most of these parameters are explained in more detail in `this blog post
        <https://huggingface.co/blog/how-to-generate>`__.

        Parameters:

            input_ids (:obj:`tf.Tensor` of :obj:`dtype=tf.int32` and shape :obj:`(batch_size, sequence_length)`, `optional`):
                The sequence used as a prompt for the generation. If :obj:`None` the method initializes it as an empty
                :obj:`tf.Tensor` of shape :obj:`(1,)`.
            max_length (:obj:`int`, `optional`, defaults to 20):
                The maximum length of the sequence to be generated.
            min_length (:obj:`int`, `optional`, defaults to 10):
                The minimum length of the sequence to be generated.
            do_sample (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to use sampling ; use greedy decoding otherwise.
            early_stopping (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether to stop the beam search when at least ``num_beams`` sentences are finished per batch or not.
            num_beams (:obj:`int`, `optional`, defaults to 1):
                Number of beams for beam search. 1 means no beam search.
            temperature (:obj:`float`, `optional`, defaults to 1.0):
                The value used to module the next token probabilities.
            top_k (:obj:`int`, `optional`, defaults to 50):
                The number of highest probability vocabulary tokens to keep for top-k-filtering.
            top_p (:obj:`float`, `optional`, defaults to 1.0):
                If set to float < 1, only the most probable tokens with probabilities that add up to ``top_p`` or
                higher are kept for generation.
            repetition_penalty (:obj:`float`, `optional`, defaults to 1.0):
                The parameter for repetition penalty. 1.0 means no penalty. See `this paper
                <https://arxiv.org/pdf/1909.05858.pdf>`__ for more details.
            pad_token_id (:obj:`int`, `optional`):
                The id of the `padding` token.
            bos_token_id (:obj:`int`, `optional`):
                The id of the `beginning-of-sequence` token.
            eos_token_id (:obj:`int`, `optional`):
                The id of the `end-of-sequence` token.
            length_penalty (:obj:`float`, `optional`, defaults to 1.0):
                Exponential penalty to the length. 1.0 means no penalty.

                Set to values < 1.0 in order to encourage the model to generate shorter sequences, to a value > 1.0 in
                order to encourage the model to produce longer sequences.
            no_repeat_ngram_size (:obj:`int`, `optional`, defaults to 0):
                If set to int > 0, all ngrams of that size can only occur once.
            bad_words_ids(:obj:`List[int]`, `optional`):
                List of token ids that are not allowed to be generated. In order to get the tokens of the words that
                should not appear in the generated text, use :obj:`tokenizer.encode(bad_word, add_prefix_space=True)`.
            num_return_sequences(:obj:`int`, `optional`, defaults to 1):
                The number of independently computed returned sequences for each element in the batch.
            attention_mask (:obj:`tf.Tensor` of :obj:`dtype=tf.int32` and shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values are in ``[0, 1]``, 1 for
                tokens that are not masked, and 0 for masked tokens.

                If not provided, will default to a tensor the same shape as :obj:`input_ids` that masks the pad token.

                `What are attention masks? <../glossary.html#attention-mask>`__
            decoder_start_token_id (:obj:`int`, `optional`):
                If an encoder-decoder model starts decoding with a different token than `bos`, the id of that token.
            use_cache: (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether or not the model should use the past last key/values attentions (if applicable to the model) to
                speed up decoding.
            model_specific_kwargs:
                Additional model specific kwargs will be forwarded to the :obj:`forward` function of the model.

        Return:

            :obj:`tf.Tensor` of :obj:`dtype=tf.int32` and shape :obj:`(batch_size * num_return_sequences,
            sequence_length)`: The generated sequences. The second dimension (sequence_length) is either equal to
            :obj:`max_length` or shorter if all batches finished early due to the :obj:`eos_token_id`.

        Examples::

            tokenizer = AutoTokenizer.from_pretrained('distilgpt2')   # Initialize tokenizer
            model = TFAutoModelWithLMHead.from_pretrained('distilgpt2')    # Download model and configuration from huggingface.co and cache.
            outputs = model.generate(max_length=40)  # do greedy decoding
            print('Generated: {}'.format(tokenizer.decode(outputs[0], skip_special_tokens=True)))

            tokenizer = AutoTokenizer.from_pretrained('openai-gpt')   # Initialize tokenizer
            model = TFAutoModelWithLMHead.from_pretrained('openai-gpt')    # Download model and configuration from huggingface.co and cache.
            input_context = 'The dog'
            input_ids = tokenizer.encode(input_context, return_tensors='tf')  # encode input context
            outputs = model.generate(input_ids=input_ids, num_beams=5, num_return_sequences=3, temperature=1.5)  # generate 3 independent sequences using beam search decoding (5 beams) with sampling from initial context 'The dog'
            for i in range(3): #  3 output sequences were generated
                print('Generated {}: {}'.format(i, tokenizer.decode(outputs[i], skip_special_tokens=True)))

            tokenizer = AutoTokenizer.from_pretrained('distilgpt2')   # Initialize tokenizer
            model = TFAutoModelWithLMHead.from_pretrained('distilgpt2')    # Download model and configuration from huggingface.co and cache.
            input_context = 'The dog'
            input_ids = tokenizer.encode(input_context, return_tensors='tf')  # encode input context
            outputs = model.generate(input_ids=input_ids, max_length=40, temperature=0.7, num_return_sequences=3, do_sample=True)  # generate 3 candidates using sampling
            for i in range(3): #  3 output sequences were generated
                print('Generated {}: {}'.format(i, tokenizer.decode(outputs[i], skip_special_tokens=True)))

            tokenizer = AutoTokenizer.from_pretrained('ctrl')   # Initialize tokenizer
            model = TFAutoModelWithLMHead.from_pretrained('ctrl')    # Download model and configuration from huggingface.co and cache.
            input_context = 'Legal My neighbor is'  # "Legal" is one of the control codes for ctrl
            input_ids = tokenizer.encode(input_context, return_tensors='tf')  # encode input context
            outputs = model.generate(input_ids=input_ids, max_length=50, temperature=0.7, repetition_penalty=1.2)  # generate sequences
            print('Generated: {}'.format(tokenizer.decode(outputs[0], skip_special_tokens=True)))

            tokenizer = AutoTokenizer.from_pretrained('gpt2')   # Initialize tokenizer
            model = TFAutoModelWithLMHead.from_pretrained('gpt2')    # Download model and configuration from huggingface.co and cache.
            input_context = 'My cute dog'
            bad_words_ids = [tokenizer.encode(bad_word, add_prefix_space=True) for bad_word in ['idiot', 'stupid', 'shut up']]
            input_ids = tokenizer.encode(input_context, return_tensors='tf')  # encode input context
            outputs = model.generate(input_ids=input_ids, max_length=100, do_sample=True, bad_words_ids=bad_words_ids)  # generate sequences without allowing bad_words to be generated
        """
        ...
    
    def adjust_logits_during_generation(self, logits, **kwargs):
        """
        Implement in subclasses of :class:`~transformers.PreTrainedModel` for custom behavior to adjust the logits in
        the generate method.
        """
        ...
    


def calc_banned_ngram_tokens(prev_input_ids, num_hypos, no_repeat_ngram_size, cur_len):
    ...

def calc_banned_bad_words_ids(prev_input_ids, bad_words_ids):
    ...

def tf_top_k_top_p_filtering(logits, top_k=..., top_p=..., filter_value=..., min_tokens_to_keep=...):
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering

    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    ...

def scatter_values_on_batch_indices(values, batch_indices):
    ...

def set_tensor_by_indices_to_value(tensor, indices, value):
    ...

def sample_without_replacement(logits, num_samples):
    """
    categorical sampling without replacement is currently not implemented the gumbel-max trick will do for now see
    https://github.com/tensorflow/tensorflow/issues/9260 for more info
    """
    ...

def shape_list(x):
    """Deal with dynamic shape in tensorflow cleanly."""
    ...

class BeamHypotheses(object):
    def __init__(self, num_beams, max_length, length_penalty, early_stopping) -> None:
        """
        Initialize n-best list of hypotheses.
        """
        ...
    
    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        ...
    
    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        ...
    
    def is_done(self, best_sum_logprobs, cur_len):
        """
        If there are enough hypotheses and that none of the hypotheses being generated can become better than the worst
        one in the heap, then we are done with this sentence.
        """
        ...
    



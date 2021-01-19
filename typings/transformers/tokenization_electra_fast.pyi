"""
This type stub file was generated by pyright.
"""

from .tokenization_bert_fast import BertTokenizerFast

"""
This type stub file was generated by pyright.
"""
VOCAB_FILES_NAMES = { "vocab_file": "vocab.txt","tokenizer_file": "tokenizer.json" }
PRETRAINED_VOCAB_FILES_MAP = { "vocab_file": { "google/electra-small-generator": "https://s3.amazonaws.com/models.huggingface.co/bert/google/electra-small-generator/vocab.txt","google/electra-base-generator": "https://s3.amazonaws.com/models.huggingface.co/bert/google/electra-base-generator/vocab.txt","google/electra-large-generator": "https://s3.amazonaws.com/models.huggingface.co/bert/google/electra-large-generator/vocab.txt","google/electra-small-discriminator": "https://s3.amazonaws.com/models.huggingface.co/bert/google/electra-small-discriminator/vocab.txt","google/electra-base-discriminator": "https://s3.amazonaws.com/models.huggingface.co/bert/google/electra-base-discriminator/vocab.txt","google/electra-large-discriminator": "https://s3.amazonaws.com/models.huggingface.co/bert/google/electra-large-discriminator/vocab.txt" },"tokenizer_file": { "google/electra-small-generator": "https://s3.amazonaws.com/models.huggingface.co/bert/google/electra-small-generator/tokenizer.json","google/electra-base-generator": "https://s3.amazonaws.com/models.huggingface.co/bert/google/electra-base-generator/tokenizer.json","google/electra-large-generator": "https://s3.amazonaws.com/models.huggingface.co/bert/google/electra-large-generator/tokenizer.json","google/electra-small-discriminator": "https://s3.amazonaws.com/models.huggingface.co/bert/google/electra-small-discriminator/tokenizer.json","google/electra-base-discriminator": "https://s3.amazonaws.com/models.huggingface.co/bert/google/electra-base-discriminator/tokenizer.json","google/electra-large-discriminator": "https://s3.amazonaws.com/models.huggingface.co/bert/google/electra-large-discriminator/tokenizer.json" } }
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = { "google/electra-small-generator": 512,"google/electra-base-generator": 512,"google/electra-large-generator": 512,"google/electra-small-discriminator": 512,"google/electra-base-discriminator": 512,"google/electra-large-discriminator": 512 }
PRETRAINED_INIT_CONFIGURATION = { "google/electra-small-generator": { "do_lower_case": True },"google/electra-base-generator": { "do_lower_case": True },"google/electra-large-generator": { "do_lower_case": True },"google/electra-small-discriminator": { "do_lower_case": True },"google/electra-base-discriminator": { "do_lower_case": True },"google/electra-large-discriminator": { "do_lower_case": True } }
class ElectraTokenizerFast(BertTokenizerFast):
    r"""
    Construct a "fast" ELECTRA tokenizer (backed by HuggingFace's `tokenizers` library).

    :class:`~transformers.ElectraTokenizerFast` is identical to :class:`~transformers.BertTokenizerFast` and runs
    end-to-end tokenization: punctuation splitting and wordpiece.

    Refer to superclass :class:`~transformers.BertTokenizerFast` for usage examples and documentation concerning
    parameters.
    """
    vocab_files_names = ...
    pretrained_vocab_files_map = ...
    max_model_input_sizes = ...
    pretrained_init_configuration = ...
    slow_tokenizer_class = ...



"""
This type stub file was generated by pyright.
"""

from .tokenization_bert import BertTokenizer

VOCAB_FILES_NAMES = { "vocab_file": "vocab.txt" }
PRETRAINED_VOCAB_FILES_MAP = { "vocab_file": { "google/electra-small-generator": "https://s3.amazonaws.com/models.huggingface.co/bert/google/electra-small-generator/vocab.txt","google/electra-base-generator": "https://s3.amazonaws.com/models.huggingface.co/bert/google/electra-base-generator/vocab.txt","google/electra-large-generator": "https://s3.amazonaws.com/models.huggingface.co/bert/google/electra-large-generator/vocab.txt","google/electra-small-discriminator": "https://s3.amazonaws.com/models.huggingface.co/bert/google/electra-small-discriminator/vocab.txt","google/electra-base-discriminator": "https://s3.amazonaws.com/models.huggingface.co/bert/google/electra-base-discriminator/vocab.txt","google/electra-large-discriminator": "https://s3.amazonaws.com/models.huggingface.co/bert/google/electra-large-discriminator/vocab.txt" } }
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = { "google/electra-small-generator": 512,"google/electra-base-generator": 512,"google/electra-large-generator": 512,"google/electra-small-discriminator": 512,"google/electra-base-discriminator": 512,"google/electra-large-discriminator": 512 }
PRETRAINED_INIT_CONFIGURATION = { "google/electra-small-generator": { "do_lower_case": True },"google/electra-base-generator": { "do_lower_case": True },"google/electra-large-generator": { "do_lower_case": True },"google/electra-small-discriminator": { "do_lower_case": True },"google/electra-base-discriminator": { "do_lower_case": True },"google/electra-large-discriminator": { "do_lower_case": True } }
class ElectraTokenizer(BertTokenizer):
    r"""
    Construct an ELECTRA tokenizer.

    :class:`~transformers.ElectraTokenizer` is identical to :class:`~transformers.BertTokenizer` and runs end-to-end
    tokenization: punctuation splitting and wordpiece.

    Refer to superclass :class:`~transformers.BertTokenizer` for usage examples and documentation concerning
    parameters.
    """
    vocab_files_names = ...
    pretrained_vocab_files_map = ...
    max_model_input_sizes = ...
    pretrained_init_configuration = ...


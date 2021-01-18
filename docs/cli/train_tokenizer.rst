Train a new tokenizer on text files
=======================================================================================================================

Training a new tokenizer is a 2 step process. First, you train a new fast tokenizer (:class:`~tokenizers.Tokenizer`)
using `🤗/tokenizers <https://github.com/huggingface/tokenizers>`__ library. Second, you convert the pretrained tokenizer
into :class:`~transformers.PreTrainedTokenizerFast` and save the tokenizer files. From now on, you can load saved
pretrained tokenizers for data processing and model training. Our cli provides a `train_tokenizer` subcommand to train 
a new tokenizer.

All you have to do is to make or use a built-in tokenizer trainer (:class:`~formerbox.TokenizerTrainer`) class,
and prepare an instance with its defined dataclass params (i.e. pass arguments through a cli command).

Subcommand required parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: formerbox.cli.TrainTokenizer.Params
    :members:

Built-in tokenizer trainers in the library
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These are the built-in :class:`~formerbox.TokenizerTrainer` components you can use to train a tokenizer.

gpt2
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Trains a :class:`~tokenizers.ByteLevelBPETokenizer` and then converts it to :class:`~formerbox.GPT2Tokenizer`.

Required parameters
***********************************************************************************************************************

.. autoclass:: formerbox.GPT2TokenizerTrainer.Params
    :members:

.. autoclass:: formerbox.data.tokenizers.TokenizerTrainerParams
    :members:

roberta
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Trains a :class:`~tokenizers.ByteLevelBPETokenizer` and then converts it to :class:`~formerbox.RobertaTokenizer`.

Required parameters
***********************************************************************************************************************

.. autoclass:: formerbox.RobertaTokenizerTrainer.Params
    :members:

.. autoclass:: formerbox.data.tokenizers.TokenizerTrainerParams
    :members:

bart
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Trains a :class:`~tokenizers.ByteLevelBPETokenizer` and then converts it to :class:`~formerbox.BartTokenizer`.

Required parameters
***********************************************************************************************************************

.. autoclass:: formerbox.BartTokenizerTrainer.Params
    :members:

.. autoclass:: formerbox.data.tokenizers.TokenizerTrainerParams
    :members:

Example cli command
***********************************************************************************************************************

Note, that transformer-based models have sequence length limits. That's why you probably need to set max length in advance.
Basically, your tokenizer max length should match a transformer model's max length value.

.. code-block:: shell

    formerbox-cli train_tokenizer                   \
            --tokenizer roberta                     \
            --save_directory <path>                 \
            --files <text_file>[<text_file>...]     \
            --vocab_size <vocab_size>               \
            --model_max_length <max_length>

Making your own tokenizer trainer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If no built-in component fits to your needs you can make a new one based on the :class:`~formerbox.TokenizerTrainer` class.
Note, that we provide some out-of-the-box features in :class:`~formerbox.TokenizerTrainerBase` class, so you can inherit
from it directly. You'll need to define a backend :class:`~tokenizers.Tokenizer` and implement abstract methods.

You will also need a custom :class:`~transformers.PreTrainedTokenizerFast` class to convert backend tokenizer. Consider
reusing already existing tokenizers from `🤗/transformers <https://github.com/huggingface/transformers>`__ library.

.. autoclass:: formerbox.TokenizerTrainer
    :members:

.. autoclass:: formerbox.TokenizerTrainerBase
    :members:

.. autoclass:: transformers.PreTrainedTokenizerFast
    :members:

Example of a new tokenizer trainer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from dataclasses import dataclass, field
    from pathlib import Path
    from typing import Any, Dict, List, Optional, Text, Union

    from tokenizers.implementations import ByteLevelBPETokenizer
    from transformers import PreTrainedTokenizerFast
    from formerbox.common.dataclass_argparse import DataclassBase
    from formerbox.tasks.tokenization_trainer import TokenizerTrainerBase


    @TokenizerTrainer.register(name="my_tokenizer")
    class MyTokenizerTrainer(TokenizerTrainerBase):
        @dataclass
        class Params(DataclassBase):
            ### Your fields here

        params: Params
        params_type = Params

        def __init__(self, params: Params, **kwargs: Any) -> None:
            super().__init__(params, **kwargs)

        @classmethod
        def build_tokenizer(cls, params: Params) -> ByteLevelBPETokenizer:
            ### Create and return an instance of `ByteLevelBPETokenizer`
            return ByteLevelBPETokenizer(...)

        def configure_tokenizer(
            self, tokenizer_path: Union[Text, Path], **kwargs: Any
        ) -> PreTrainedTokenizerFast:
            ### Configure an instance of `PreTrainedTokenizerFast`

        def train_tokenizer(self, *args: Any, **kwargs: Any) -> None:
            ### Train the backend tokenizer

        def save_pretrained(
            self, save_directory: Text, legacy_format: bool, **kwargs: Any
        ) -> None:
            super().save_pretrained(
                save_directory,
                legacy_format,
                **kwargs,
            )


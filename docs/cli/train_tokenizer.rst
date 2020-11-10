Train a new tokenizer on text files
=======================================================================================================================

Training a new tokenizer takes 2 steps. First, you train a new fast tokenizer (:class:`~tokenizers.Tokenizer`) using 
`🤗/tokenizers <https://github.com/huggingface/tokenizers>`__ library. Second, you convert the pretrained tokenizer
into :class:`~transformers.PreTrainedTokenizerFast` and save the tokenizer files. From now on, you can load saved
pretrained tokenizers for data processing and model training. Our cli provides a `train_tokenizer` subcommand to train 
a new tokenizer.

All you have to do is to make or use a built-in tokenizer trainer (:class:`~formerbox.TokenizerTrainer`) class,
and prepare an instance with its defined dataclass params (i.e. pass arguments through a cli command).

Built-in tokenizer trainers in the library
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These are the built-in :class:`~formerbox.TokenizerTrainer` components you can use to train a tokenizer.

roberta
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Trains a :class:`~tokenizers.ByteLevelBPETokenizer` and then converts it to :class:`~formerbox.RobertaTokenizer`.

Required parameters
***********************************************************************************************************************

.. autoclass:: formerbox.cli.TrainTokenizer.Params
    :members:

.. autoclass:: formerbox.RobertaTokenizerTrainer.Params
    :members:

Example cli command
***********************************************************************************************************************

.. code-block:: shell

    python -m formerbox train_tokenizer             \
            --tokenizer roberta                     \
            --save_directory <path>                 \
            --files <text_file>[<text_file>...]     \
            --vocab_size <vocab_size>

Making your own tokenizer trainer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If no built-in component fits to your needs you can make a new one based on the :class:`~formerbox.TokenizerTrainer` class.
Note, that we provide some out-of-the-box features in :class:`~formerbox.BaseTokenizerTrainer` class, so you can inherit
from it directly. You'll need to define a backend :class:`~tokenizers.Tokenizer` and implement abstract methods.

.. autoclass:: formerbox.TokenizerTrainer
    :members:

.. autoclass:: formerbox.BaseTokenizerTrainer
    :members:

Example of a new tokenizer trainer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import logging
    from dataclasses import dataclass, field
    from pathlib import Path
    from typing import Any, Dict, List, Optional, Text, Union

    from tokenizers.implementations import ByteLevelBPETokenizer
    from transformers import PreTrainedTokenizerFast
    from formerbox.common.dataclass_argparse import DataclassBase
    from formerbox.tasks.tokenization_trainer import BaseTokenizerTrainer

    logger = logging.getLogger(__name__)


    @BaseTokenizerTrainer.register(name="my-tokenizer", constructor="from_partial")
    class MyTokenizerTrainer(BaseTokenizerTrainer):
        # pylint: disable=arguments-differ
        @dataclass
        class Params(DataclassBase):
            ...

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


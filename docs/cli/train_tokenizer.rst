Train a new tokenizer on text files
=======================================================================================================================

You can train a new tokenizer (:class:`~tokenizers.Tokenizer`) and convert it to either
:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast` instance with the
`train_tokenizer` cli subcommand.

All you have to do is to make or use a built-in tokenizer module (:class:`~gitnetic.TokenizerModule`) class,
and prepare an instance with its defined dataclass params (i.e. pass arguments through a cli command).

Built-in tokenizer modules in the library
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These are the built-in :class:`~gitnetic.TokenizerModule` components you can use to train a tokenizer.

transformer-tokenizer-fast
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Trains a :class:`~tokenizers.ByteLevelBPETokenizer` and then converts it to :class:`~TransformerTokenizerFast`.

Required parameters
***********************************************************************************************************************

.. autoclass:: gitnetic.cli.TrainTokenizer.Params
    :members:

.. autoclass:: gitnetic.TransformerTokenizerModule.Params
    :members:

Example cli command
***********************************************************************************************************************

.. code-block:: shell

    python -m gitnetic train_tokenizer              \
            --tokenizer transformer-tokenizer-fast  \
            --tokenizer_path <>                     \
            --files <text_file>[<text_file>...]     \
            --vocab_size <vocab_size>

Making your own tokenizer module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If no built-in component fits to your needs you can make a new one based on the :class:`~gitnetic.TokenizerModule` class. 
You'll need to define a backend :class:`~tokenizers.Tokenizer` and implement abstract methods.

.. autoclass:: gitnetic.TokenizerModule
    :members:

Example of a new task
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from dataclasses import dataclass, field
    from pathlib import Path
    from typing import Any, List, Text, Union

    from gitneti import TokenizerModule
    from gitnetic.common.dataclass_argparse import DataclassBase
    from tokenizers import AddedToken
    from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

    Token = Union[Text, AddedToken]
    Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]


    @TokenizerModule.register(name="transformer-tokenizer-fast", constructor="from_partial")
    class TransformerTokenizerModule(TokenizerModule):
        # pylint: disable=arguments-differ
        @dataclass
        class Params(DataclassBase):
            # <YOUR ARGS HERE>

        params: Params
        params_type = Params

        def __init__(self, params: Params, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.params = params
            self.special_tokens: List[Token] = # Define your special tokens (e.g. <mask>)
            self.backend_tokenizer = ### Initialize the backend tokenizer

        def configure_tokenizer(
            self, tokenizer_path: Union[Text, Path], **kwargs: Any
        ) -> Tokenizer:
            ### Configure an instance of `PreTrainedTokenizer` or `PreTrainedTokenizerFast`

        def train_tokenizer(self, *args: Any, **kwargs: Any) -> None:
            ### Train the tokenizer

        def save_pretrained(self, *args: Any, **kwargs: Any) -> None:
            ### Step 1: Save the pre-trained backend tokenizer
            ### Step 2: Configure the tokenizer with `configure_tokenizer` method
            ### Step 3: Save the converted tokenizer with `save_pretrained` method

        @staticmethod
        def from_pretrained(params: Params, **kwargs: Any) -> Tokenizer:
            ### Make an instance of `PreTrainedTokenizer` or `PreTrainedTokenizerFast` with defined `Params`

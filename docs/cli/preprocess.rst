Preprocess text datasets
=======================================================================================================================

Transformer modules are designed to work with :class:`~formerbox.IndexedDataset` datasets. Such datasets are already
tokenized and can be easily accessed at any index with additional metadata (e.g. sample dimensions and lengths). This
is especially useful for more efficient and balanced sampling when we sample batches of uniform token lengths.

Preprocessing only works with text datasets. That means that if you have an arbitrary dataset (e.g. JSON or CSV file),
you'll have to first convert the raw dataset into its textual representation. Also note, that indexed datasets
(:class:`~formerbox.IndexedDataset`) do not support multicolumn processing, so you'll have to keep as many binarized
pair of files as many columns you want to get.

At its core preprocessing requires 2 components – :class:`~formerbox.TokenizerModule` and :class:`~formerbox.Binarizer`.

* The :class:`~formerbox.TokenizerModule` uses the pre-trained tokenizer to convert text samples into input ids 
  (i.e. token ids).

* The :class:`~formerbox.Binarizer` prepares the dataset for mapping and then maps the samples into binarized data.

Subcommand required parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: formerbox.cli.Preprocess.Params
    :members:

Built-in tokenizer modules in the library
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

byte-level-bpe-tokenizer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Uses the :class:`~formerbox.ByteLevelBPETokenizerFast` under the hood to map text samples into a tokenized encoding.

Required parameters
***********************************************************************************************************************

.. autoclass:: formerbox.ByteLevelBPETokenizerModule.Params
    :members:

Built-in binarizers in the library
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

transformer-binarizer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Converts text samples into a tokenized encoding, supports the overflowing batches with stride for long docs processing.

Required parameters
***********************************************************************************************************************

.. autoclass:: formerbox.TransformerBinarizer.Params
    :members:

Example cli command
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: shell

    python -m formerbox preprocess                          \
        --tokenizer byte-level-bpe-tokenizer                \
        --binarizer transformer-binarizer                   \
        --output_path <output_path>                         \
        --train_prefix <train_prefix>                       \
        --valid_prefix <valid_prefix>                       \
        --test_prefix <test_prefix>                         \
                                                            \
        --tokenizer_path <tokenizer_path>                   \
                                                            \
        --max_length 512 --return_overflowing_tokens true   \
        --batch_size 512 --batched true
        --num_proc 16                                       \
        --block_size 1073741824                             \
                                                            \
        --dataset_impl mmap

Making your own binarizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from dataclasses import dataclass, field
    from typing import Any, Callable, Dict, Text

    import torch
    from formerbox import Binarizer
    from formerbox.common.dataclass_argparse import DataclassBase
    from formerbox.data.indexed_dataset_setup import IndexedDatasetSetup
    from transformers import PreTrainedTokenizerFast


    @Binarizer.register(name="my-binarizer", constructor="from_partial")
    class MyBinarizer(Binarizer):
        @dataclass
        class Params(DataclassBase):
            ### Your fields here

        params: Params
        params_type = Params

        def __init__(
            self,
            dataset_setup: IndexedDatasetSetup,
            tokenizer: PreTrainedTokenizerFast,
            params: Params,
        ) -> None:
            super().__init__(dataset_setup, tokenizer)
            self.params = params

        def binarize(
            self, filename: Text, consumer: Callable[[torch.Tensor], None]
        ) -> None:
            ### Step 1: do anything to prepare the dataset for mapping
            ### Step 2: map the dataset with `self.encode` method
            ### Step 3: iterate over the processed dataset and write results to the consumer

        def encode(self, instance: Dict[Text, Any]) -> Dict[Text, Any]:
            ### Do anything to map a dataset instance into input ids

Preprocess text datasets
=======================================================================================================================

Transformer modules are designed to work with :class:`~formerbox.IndexedDatasetBase` datasets. Such datasets are already
tokenized and can be easily accessed at any index with additional metadata (e.g. sample dimensions and lengths). This
is especially useful for more efficient and balanced sampling when we sample batches of uniform token lengths.

Preprocessing only works with text datasets. That means that if you have an arbitrary dataset (e.g. JSON or CSV file),
you'll have to first convert the raw dataset into its textual representation. Also note, that indexed datasets
(:class:`~formerbox.IndexedDatasetBase`) do not support multicolumn processing, so you'll have to keep as many binarized
pair of files as many columns you want to get.

Preprocessing requires 2 components – :class:`~transformers.PreTrainedTokenizerFast` and :class:`~formerbox.Binarizer`.

* The :class:`~transformers.PreTrainedTokenizerFast` converts text samples into input ids (i.e. token ids).
* The :class:`~formerbox.Binarizer` prepares the dataset for mapping and then maps the samples into binarized data.

Subcommand required parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: formerbox.cli.Preprocess.Params
    :members:

Built-in binarizers in the library
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

default
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Converts text samples into a tokenized encoding, supports the overflowing batches with stride for long docs processing.

Required parameters
***********************************************************************************************************************

.. autoclass:: formerbox.DefaultBinarizer.Params
    :members:

Example cli command
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: shell

    formerbox-cli preprocess                                \
        --train_prefix <train_prefix>                       \
        --valid_prefix <valid_prefix>                       \
        --test_prefix <test_prefix>                         \
        --output_path <output_path>                         \
                                                            \
        --tokenizer <tokenizer>                             \
        --tokenizer_path <tokenizer_path>                   \
                                                            \
        --binarizer default                                 \
        --return_overflowing_tokens true                    \
        --batch_size 512                                    \
        --batched true                                      \
        --num_proc 16                                       \
                                                            \
        --dataset_impl mmap

translation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Converts monolingual and bilingual samples into tokenized encodings. supports the overflowing batches with stride
for long docs processing.

Required parameters
***********************************************************************************************************************

.. autoclass:: formerbox.TranslationBinarizer.Params
    :members:

Example cli command
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: shell

    formerbox-cli preprocess                                \
        --train_prefix <train_prefix>                       \
        --valid_prefix <valid_prefix>                       \
        --test_prefix <test_prefix>                         \
        --output_path <output_path>                         \
                                                            \
        --tokenizer <tokenizer>                             \
        --tokenizer_path <tokenizer_path>                   \
                                                            \
        --binarizer translation                             \
        --src_lang <src_lang>                               \
        --tgt_lang <tgt_lang>                               \
        --return_overflowing_tokens true                    \
        --batch_size 512                                    \
        --batched true                                      \
        --num_proc 16                                       \
                                                            \
        --dataset_impl mmap

Making your own binarizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from dataclasses import dataclass
    from typing import Any, Dict, List, Optional, Text, Type, Union

    from formerbox.data.binarizer import Binarizer, BinarizerBase
    from formerbox.data.indexed_dataset import IndexedDatasetBuilderBase
    from formerbox.data.indexed_dataset_setup import IndexedDatasetSetup
    from transformers import PreTrainedTokenizerFast


    @Binarizer.register(name="my_binarizer")
    class MyBinarizer(BinarizerBase):
        @dataclass
        class Params(BinarizerBase.Params):
            ### Your fields here

        params: Params
        params_type: Type[Params] = Params

        def __init__(
            self,
            dataset_setup: IndexedDatasetSetup,
            tokenizer: PreTrainedTokenizerFast,
            params: Params,
        ) -> None:
            super().__init__(dataset_setup, tokenizer, params)
            self.params = params

        def binarize_dataset(
            self,
            filename: Text,
            output_prefix: Text,
            **kwargs: Any,
        ) -> None:
            ### Step 1: prepare dataset files to binarize
            ### Step 2: binarize prepared datasets
            ### Step 3: finalize results

        def binarize(
            self,
            filename: Text,
            consumer: IndexedDatasetBuilderBase,
            **kwargs: Any,
        ) -> None:
            ### Step 1: prepare the given dataset for processing
            ### Step 2: run dataset processing to encode texts into input ids
            ### Step 3: iterate over the processed dataset and write results to the consumer

        def encode(self, instance: Dict[Text, Any]) -> Dict[Text, Any]:
            ### Map a dataset instance into input ids

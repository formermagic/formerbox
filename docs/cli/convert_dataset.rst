Convert arbitrary datasets into text datasets
=======================================================================================================================

If you want to train a transformer-based model on your arbitrary dataset, first you need to convert the dataset to a
text file. This is done with :class:`~formerbox.cli.ConvertDataset` subcommand which uses the 
:class:`~formerbox.DatasetConverter` to map any dataset into the text file for further processing.

We recommend using `🤗/datasets <https://github.com/huggingface/datasets>`__ for any mapping and processing.

Subcommand required parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: formerbox.cli.ConvertDataset.Params
    :members:

Built-in dataset converters in the library
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

default
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Converts arbitrary dataset with mapping representation containing text into the pretokenized text data files.

Required parameters
***********************************************************************************************************************

.. autoclass:: formerbox.DefaultDatasetConverter.Params
    :members:

code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Converts the github bigquery dataset into the pretokenized text data files.

Required parameters
***********************************************************************************************************************

.. autoclass:: formerbox.CodeDatasetConverter.Params
    :members:

Example cli command
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: shell

    python -m formerbox convert_dataset     \
        --converter code                    \
        --script_path json                  \
        --data_files <data_files>           \
        --output_path <output_path>         \
        --batch_size <batch_size>           \
        --num_proc <num_proc>               \
        --train_test_split true

Making your own dataset converter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from dataclasses import dataclass, field
    from typing import Dict, Text, Union

    from datasets import Dataset, DatasetDict, load_dataset
    from formerbox.common.dataclass_argparse import DataclassBase


    @DatasetConverter.register("my-converter", constructor="from_partial")
    class MyDatasetConverter(DatasetConverter, DatasetProcessingMixin):
        @dataclass
        class Params(DataclassBase):
            ### Your fields here

        params: Params
        params_type = Params

        def __init__(self, params: Params) -> None:
            self.params = params

        def convert(self, *args: Any, **kwargs: Any) -> None:
            ### Step 1: prepare the dataset for converting (e.g. deduplicate samples)
            ### Step 2: map the raw dataset into a text dataset with `self.encode` method
            ### Step 3: save the mapped dataset with `self.save_dataset` method

        def encode(self, instance: Dict[Text, Any]) -> Dict[Text, Any]:
            ### Map the raw dataset instance into a text sample

        def save_dataset(
            self, dataset: Union[Dataset, DatasetDict], output_path: Text
        ) -> None:
            ### Save the converted dataset into a text file

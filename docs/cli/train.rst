Train a transformer-based model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can train a transformer-based model with a `train` cli subcommand. All you have to do is to make or use a built-in 
:class:`~gitnetic.TaskModule` class and prepare the :class:`~gitnetic.TransformerTrainer` instance. Each of these 
components specify the required parameters in the params dataclasses (see `params_type` property for the type), 
so you will not miss one.

Built-in tasks in the library
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These are the built-in :class:`~gitnetic.TaskModule` components you can use to train a model.

transformer-task
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

The task for training transformer language models with a pre-trained tokenizer.

Required parameters
***********************************************************************************************************************

.. autoclass:: gitnetic.TaskModule.Params
    :members:

.. autoclass:: gitnetic.TransformerModule.Params
    :members:

.. autoclass:: gitnetic.TransformerDataModule.Params
    :members:

.. autoclass:: gitnetic.TransformerTrainer.Params
    :members:


See `pytorch_lightning.Trainer <https://pytorch-lightning.readthedocs.io/en/latest/trainer.html#trainer-class-api>`__ 
docs to find other :class:`~gitnetic.TransformerTrainer` parameters.


Example cli command
***********************************************************************************************************************

.. code-block:: shell

    python -m gitnetic train                                    \
        --task transformer-task                                 \
        --config_path <model-config.yml>                        \
        --tokenizer_path <tokenizer_path>                       \
                                                                \
        --warmup_steps <warmup_steps>                           \
        --learning_rate <learning_rate>                         \
        --power <power>                                         \
                                                                \
        --train_data_prefix <train_data_prefix>                 \
        --val_data_prefix <val_data_prefix>                     \
        --max_tokens <max_tokens>                               \
        --num_workers <num_workers>                             \
                                                                \
        --gpus <gpus>                                           \
        --max_steps <max_steps>                                 \
        --val_check_interval <val_check_interval>               \
        --save_step_frequency <save_step_frequency>             \
        --save_dir <save_dir>                                   \
        --progress_bar_refresh_rate <progress_bar_refresh_rate> \
        --row_log_interval <row_log_intervala

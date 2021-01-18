Train a transformer-based model
=======================================================================================================================

You can train a transformer-based model with a `train` cli subcommand. All you have to do is to make or use a built-in
:class:`~formerbox.TaskModule` class and prepare the :class:`~formerbox.TransformerTrainer` instance. Each of these 
components specify the required parameters in the params dataclasses (see `params_type` property for the type), 
so you will not miss one.

Built-in tasks in the library
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These are the built-in :class:`~formerbox.TaskModule` components you can use to train a model.

masked_lm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This task is suitable to train models with a masked language modeling objective (e.g. RoBERTa, BET, etc).

Required parameters
***********************************************************************************************************************

.. autoclass:: formerbox.MaskedLMTask.Params
    :members:

.. autoclass:: formerbox.MaskedLMModule.Params
    :members:

.. autoclass:: formerbox.MaskedLMDataModule.Params
    :members:

.. autoclass:: formerbox.TransformerTrainer.Params
    :members:

See `pytorch_lightning.Trainer <https://pytorch-lightning.readthedocs.io/en/latest/trainer.html#trainer-class-api>`__ 
docs to find other :class:`~formerbox.TransformerTrainer` parameters.

Example cli command
***********************************************************************************************************************

.. code-block:: shell

    formerbox-cli train                                         \
        --task masked_lm                                        \
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
        --log_every_n_steps <log_every_n_steps>                 \
        --wandb_project <wandb_projec>                          \
        --wandb_name <wandb_name>

translation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This task is suitable to train models for bilingual neural machine translation (compatible with seq2seq models).

Required parameters
***********************************************************************************************************************

.. autoclass:: formerbox.TranslationTask.Params
    :members:

.. autoclass:: formerbox.TranslationModule.Params
    :members:

.. autoclass:: formerbox.TranslationDataModule.Params
    :members:

.. autoclass:: formerbox.TransformerTrainer.Params
    :members:

See `pytorch_lightning.Trainer <https://pytorch-lightning.readthedocs.io/en/latest/trainer.html#trainer-class-api>`__ 
docs to find other :class:`~formerbox.TransformerTrainer` parameters.

Example cli command
***********************************************************************************************************************

.. code-block:: shell

    formerbox-cli train                                         \
        --task translation                                      \
        --config_path <model-config.yml>                        \
        --tokenizer_path <tokenizer_path>                       \
                                                                \
        --warmup_steps <warmup_steps>                           \
        --learning_rate <learning_rate>                         \
        --power <power>                                         \
        --label_smoothing <label_smoothing>                     \
                                                                \
        --train_data_prefix <train_data_prefix>                 \
        --val_data_prefix <val_data_prefix>                     \
        --src_lang <src_lang>                                   \
        --tgt_lang <tgt_lang>                                   \
        --max_tokens <max_tokens>                               \
        --num_workers <num_workers>                             \
                                                                \
        --gpus <gpus>                                           \
        --max_steps <max_steps>                                 \
        --val_check_interval <val_check_interval>               \
        --save_step_frequency <save_step_frequency>             \
        --save_dir <save_dir>                                   \
        --progress_bar_refresh_rate <progress_bar_refresh_rate> \
        --log_every_n_steps <log_every_n_steps>                 \
        --wandb_project <wandb_projec>                          \
        --wandb_name <wandb_name>

denoising
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This task is suitable to train models with a sequence denoising objective (aka BART, mBART).

Required parameters
***********************************************************************************************************************

.. autoclass:: formerbox.DenoisingTask.Params
    :members:

.. autoclass:: formerbox.DenoisingModule.Params
    :members:

.. autoclass:: formerbox.DenoisingDataModule.Params
    :members:

.. autoclass:: formerbox.TransformerTrainer.Params
    :members:

See `pytorch_lightning.Trainer <https://pytorch-lightning.readthedocs.io/en/latest/trainer.html#trainer-class-api>`__ 
docs to find other :class:`~formerbox.TransformerTrainer` parameters.

Example cli command
***********************************************************************************************************************

.. code-block:: shell

    formerbox-cli train                                         \
        --task denoising                                        \
        --config_path <model-config.yml>                        \
        --tokenizer_path <tokenizer_path>                       \
                                                                \
        --warmup_steps <warmup_steps>                           \
        --learning_rate <learning_rate>                         \
        --power <power>                                         \
        --label_smoothing <label_smoothing>                     \
                                                                \
        --train_data_prefix <train_data_prefix>                 \
        --val_data_prefix <val_data_prefix>                     \
        --src_lang <src_lang>                                   \
        --lambda_coef <lambda_coef>                             \
        --masked_token_ratio <mask_ratio>                       \
        --random_token_ratio <random_ratio>                     \
        --max_tokens <max_tokens>                               \
        --num_workers <num_workers>                             \
                                                                \
        --gpus <gpus>                                           \
        --max_steps <max_steps>                                 \
        --val_check_interval <val_check_interval>               \
        --save_step_frequency <save_step_frequency>             \
        --save_dir <save_dir>                                   \
        --progress_bar_refresh_rate <progress_bar_refresh_rate> \
        --log_every_n_steps <log_every_n_steps>                 \
        --wandb_project <wandb_projec>                          \
        --wandb_name <wandb_name>

word_lm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This task is suitable to train models with a whole word masking objective (compatible with seq2seq models).

Required parameters
***********************************************************************************************************************

.. autoclass:: formerbox.WordLMTask.Params
    :members:

.. autoclass:: formerbox.WordLMModule.Params
    :members:

.. autoclass:: formerbox.WordLMDataModule.Params
    :members:

.. autoclass:: formerbox.TransformerTrainer.Params
    :members:

See `pytorch_lightning.Trainer <https://pytorch-lightning.readthedocs.io/en/latest/trainer.html#trainer-class-api>`__ 
docs to find other :class:`~formerbox.TransformerTrainer` parameters.

Example cli command
***********************************************************************************************************************

.. code-block:: shell

    formerbox-cli train                                         \
        --task word_lm                                          \
        --config_path <model-config.yml>                        \
        --tokenizer_path <tokenizer_path>                       \
                                                                \
        --warmup_steps <warmup_steps>                           \
        --learning_rate <learning_rate>                         \
        --power <power>                                         \
        --label_smoothing <label_smoothing>                     \
                                                                \
        --train_data_prefix <train_data_prefix>                 \
        --val_data_prefix <val_data_prefix>                     \
        --lang <src_lang>                                       \
        --masked_token_ratio <mask_ratio>                       \
        --random_token_ratio <random_ratio>                     \
        --max_tokens <max_tokens>                               \
        --num_workers <num_workers>                             \
                                                                \
        --gpus <gpus>                                           \
        --max_steps <max_steps>                                 \
        --val_check_interval <val_check_interval>               \
        --save_step_frequency <save_step_frequency>             \
        --save_dir <save_dir>                                   \
        --progress_bar_refresh_rate <progress_bar_refresh_rate> \
        --log_every_n_steps <log_every_n_steps>                 \
        --wandb_project <wandb_projec>                          \
        --wandb_name <wandb_name>

Making your own task
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If no built-in task fits to your needs you can make a new one based on the :class:`~formerbox.TaskModule` class. You'll 
need to specify a module and datamodule to define a new task. Each task should also implement the :func:`~formerbox.TaskModule.setup` 
method where given the tuple with parsed params you initialize the task objects.

Example of a new task
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from argparse import Namespace
    from dataclasses import dataclass
    from typing import Tuple, Type, Union

    from formerbox.common.dataclass_argparse import DataclassArgumentParser, DataclassBase
    from formerbox.common.has_params import ParamsType
    from formerbox.tasks.task_module import TaskModule, TaskParams
    from formerbox.training.load_from_config import model_from_config, tokenizer_from_config

    from formerbox.modules import MyDataModule as DataModule
    from formerbox.modules import MyModule as Module
    from transformers import PreTrainedTokenizerFast as Tokenizer


    @TaskModule.register("my_task")
    class MyTask(TaskModule[ParamsType]):
        @dataclass
        class Params(TaskParams):
            ### Your fields here

        params: Params
        params_type: Type[Params] = Params

        def __init__(
            self,
            tokenizer: Tokenizer,
            module: Module,
            datamodule: DataModule,
        ) -> None:
            super().__init__(tokenizer, module, datamodule)
            self.tokenizer = tokenizer
            self.module = module
            self.datamodule = datamodule

        @classmethod
        def setup(
            cls: Type["MaskedLMTask"],
            params: Tuple[Union[DataclassBase, Namespace], ...],
        ) -> "MaskedLMTask":
            ### Prepare components given the provided config
            tokenizer = ...
            module = ...
            datamodule = ...

            ### Return an instance of Self with prepared components
            return cls(tokenizer, module, datamodule)

        @classmethod
        def add_argparse_params(
            cls: Type["MaskedLMTask"], parser: DataclassArgumentParser
        ) -> None:
            parser.add_arguments(cls.params_type)
            Module.add_argparse_params(parser)
            DataModule.add_argparse_params(parser)

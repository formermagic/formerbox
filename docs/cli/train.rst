Train a transformer-based model
=======================================================================================================================

You can train a transformer-based model with a `train` cli subcommand. All you have to do is to make or use a built-in
:class:`~gitnetic.TaskModule` class and prepare the :class:`~gitnetic.TransformerTrainer` instance. Each of these 
components specify the required parameters in the params dataclasses (see `params_type` property for the type), 
so you will not miss one.

Built-in tasks in the library
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These are the built-in :class:`~gitnetic.TaskModule` components you can use to train a model.

transformer-task
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

Making your own task
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If no built-in task fits to your needs you can make a new one based on the :class:`~gitnetic.TaskModule` class. You'll 
need to specify a module and datamodule to define a new task. Each task should also implement the 
:func:`~gitnetic.TaskModule.setup` method where given the tuple with parsed params you initialize the task comps.

Example of a new task
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from argparse import Namespace
    from dataclasses import dataclass, field
    from typing import Text, Tuple, Type, Union

    from gitnetic.common.dataclass_argparse import (
        DataclassArgumentParser,
        DataclassBase,
        get_params_item,
    )
    from gitnetic.tasks.base_transformers.base_task import (
        TaskModule,
        TransformerDataModule,
        TransformerModule,
        model_from_config,
        tokenizer_from_config,
    )

    ParamType = Union[DataclassBase, Namespace]


    @TaskModule.register("my-custom-task")
    class MyCustomTask(TaskModule[TransformerModule, TransformerDataModule]):
        @dataclass
        class Params(DataclassBase):
            config_path: Text = field()
            tokenizer_path: Text = field()

        params: Params
        params_type: Type[Params] = Params

        ComponentParams = Tuple[
            params_type,
            TransformerModule.params_type,
            TransformerDataModule.params_type,
        ]

        @classmethod
        def get_params(cls, params: Tuple[ParamType, ...]) -> "ComponentParams":
            # get the params for task components
            task_params = get_params_item(
                params=params,
                params_type=cls.params_type,
            )
            module_params = get_params_item(
                params=params,
                params_type=TransformerModule.params_type,
            )
            datamodule_params = get_params_item(
                params=params,
                params_type=TransformerDataModule.params_type,
            )

            # make sure the params exist
            assert isinstance(task_params, cls.params_type)
            assert isinstance(module_params, TransformerModule.params_type)
            assert isinstance(datamodule_params, TransformerDataModule.params_type)

            return task_params, module_params, datamodule_params

        @classmethod
        def setup(
            cls: Type["MyCustomTask"], params: Tuple[ParamType, ...]
        ) -> "MyCustomTask":
            # get the params for task components
            task_params, module_params, datamodule_params = cls.get_params(params)

            # prepare the tokenizer from config
            tokenizer = tokenizer_from_config(
                task_params.config_path, task_params.tokenizer_path
            )

            # prepare a model to train
            model = model_from_config(
                task_params.config_path,
                vocab_size=tokenizer.vocab_size,
                pad_token_id=tokenizer.pad_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            # prepare a transformer module
            module = TransformerModule(
                model=model, tokenizer=tokenizer, params=module_params
            )

            # prepare a transformer datamodule
            datamodule = TransformerDataModule(
                tokenizer=tokenizer, params=datamodule_params
            )

            return cls(tokenizer, module, datamodule)

        @classmethod
        def add_argparse_params(
            cls: Type["MyCustomTask"], parser: DataclassArgumentParser
        ) -> None:
            parser.add_arguments(cls.Params)
            TransformerModule.add_argparse_params(parser)
            TransformerDataModule.add_argparse_params(parser)

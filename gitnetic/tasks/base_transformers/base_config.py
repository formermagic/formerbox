from importlib import import_module
from pathlib import Path
from typing import Any, Text, Type, Union

import yaml
from transformers import PretrainedConfig, PreTrainedModel


def import_class_from_string(path: Text) -> Type:
    module_path, _, class_name = path.rpartition(".")
    mod = import_module(module_path)
    klass = getattr(mod, class_name)
    return klass


def validate_config(config: Any) -> None:
    assert isinstance(config, dict), "Config must be decoded into a dictionary"
    assert "model" in config, "Config must contain a `model` property"
    assert "name" in config["model"], "Config must contain a `model.name` property"
    assert "config" in config["model"], "Config must contain a `model.config` property"
    assert "params" in config["model"], "Config must contain `model.params` property"


def model_from_config(config_path: Union[Text, Path], **kwargs: Any) -> PreTrainedModel:
    with open(config_path, mode="r") as config_file:
        config_kwargs = yaml.load(config_file)
        validate_config(config_kwargs)
    try:
        model_name = config_kwargs["model"]["name"]
        config_name = config_kwargs["model"]["config"]
        model_class = import_class_from_string(model_name)
        config_class = import_class_from_string(config_name)
    except AttributeError as err:
        raise err

    params = config_kwargs["model"]["params"]
    params.update(kwargs)
    config = config_class(**params)
    assert isinstance(config, PretrainedConfig)

    return model_class(config)

from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Text, Type, Union

import yaml
from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizerBase


def import_class_from_string(path: Text) -> Type:
    module_path, _, class_name = path.rpartition(".")
    mod = import_module(module_path)
    klass = getattr(mod, class_name)
    return klass


def validate_model_config(config: Any) -> None:
    assert isinstance(config, dict), "Config must be decoded into a dictionary"
    assert "model" in config, "Config must contain a `model` property"
    assert "name" in config["model"], "Config must contain a `model.name` property"
    assert "config" in config["model"], "Config must contain a `model.config` property"
    assert "params" in config["model"], "Config must contain `model.params` property"


def validate_tokenizer_config(config: Any) -> None:
    assert isinstance(config, dict), "Config must be decoded into a dictionary"
    assert "tokenizer" in config, "Config must contain a `tokenizer` property"
    assert (
        "params" in config["tokenizer"]
    ), "Config must contain `tokenizer.params` property"


def parse_config(config_path: Union[Text, Path]) -> Dict[Text, Any]:
    with open(config_path, mode="r") as config_file:
        config_kwargs = yaml.load(config_file, Loader=yaml.FullLoader)
    return config_kwargs


def model_from_config(config_path: Union[Text, Path], **kwargs: Any) -> PreTrainedModel:
    try:
        config_kwargs = parse_config(config_path)
        validate_model_config(config_kwargs)
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


def tokenizer_from_config(
    config_path: Union[Text, Path], tokenizer_path: Union[Text, Path], **kwargs: Any
) -> PreTrainedTokenizerBase:
    try:
        config_kwargs = parse_config(config_path)
        validate_tokenizer_config(config_kwargs)
        tokenizer_name = config_kwargs["tokenizer"]["name"]
        tokenizer_class = import_class_from_string(tokenizer_name)
    except AttributeError as err:
        raise err

    assert issubclass(tokenizer_class, PreTrainedTokenizerBase)

    if isinstance(tokenizer_path, Path):
        tokenizer_path = tokenizer_path.as_posix()

    params = config_kwargs["tokenizer"].get("params", {})
    params.update(kwargs)
    tokenizer = tokenizer_class.from_pretrained(tokenizer_path, **params)

    return tokenizer

import logging
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Text, Type, Union

import yaml
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

logger = logging.getLogger(__name__)


def import_class_from_string(path: Text) -> Type[Any]:
    module_path, _, class_name = path.rpartition(".")
    mod = import_module(module_path)
    klass = getattr(mod, class_name)
    return klass


def validate_model_config(config: Any) -> None:
    assert isinstance(config, dict), "Config must be decoded into a dictionary"
    assert "model" in config, "Config must contain a `model` property"
    assert "name" in config["model"], "Config must contain a `model.name` property"
    assert "config" in config["model"], "Config must contain a `model.config` property"
    assert (
        "config_params" in config["model"]
    ), "Config must contain `model.config_params` property"


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

    config_params = config_kwargs["model"]["config_params"]
    config_params.update(kwargs)
    config = config_class(**config_params)
    assert isinstance(config, PretrainedConfig)

    return model_class(config)


def tokenizer_from_config(
    config_path: Union[Text, Path], tokenizer_path: Union[Text, Path], **kwargs: Any
) -> Tokenizer:
    try:
        config_kwargs = parse_config(config_path)
        validate_tokenizer_config(config_kwargs)
        tokenizer_name = config_kwargs["tokenizer"]["name"]
        tokenizer_class = import_class_from_string(tokenizer_name)
        assert issubclass(tokenizer_class, Tokenizer.__args__)  # type: ignore
    except AttributeError as err:
        raise err

    if isinstance(tokenizer_path, Path):
        tokenizer_path = tokenizer_path.as_posix()

    params = config_kwargs["tokenizer"].get("params", {})
    params.update(kwargs)
    tokenizer = tokenizer_class.from_pretrained(tokenizer_path, **params)  # type: ignore

    return tokenizer

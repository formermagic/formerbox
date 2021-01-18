import logging
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Text, Type, Union

import yaml
from formerbox.utils import update_left_inplace
from transformers import PretrainedConfig, PreTrainedModel
from transformers import PreTrainedTokenizerFast as Tokenizer

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
        "name" in config["model"]["config"]
    ), "Config must contain `model.config.name` property"


def validate_tokenizer_config(config: Any) -> None:
    assert isinstance(config, dict), "Config must be decoded into a dictionary"
    assert "tokenizer" in config, "Config must contain a `tokenizer` property"


def parse_config(config_path: Union[Text, Path]) -> Dict[Text, Any]:
    with open(config_path, mode="r") as config_file:
        config_kwargs = yaml.load(config_file, Loader=yaml.FullLoader)
    return config_kwargs


def model_from_config(config_path: Union[Text, Path], **kwargs: Any) -> PreTrainedModel:
    try:
        # parse config and make sure it contains model setup
        config_kwargs = parse_config(config_path)
        validate_model_config(config_kwargs)

        # extract model setup parameters
        model_kwargs = config_kwargs["model"]
        assert isinstance(model_kwargs, dict)
        # extract model config setup parameters
        model_config_kwargs = model_kwargs.pop("config")
        assert isinstance(model_config_kwargs, dict)

        # get the model class
        model_name = model_kwargs.pop("name")
        model_class = import_class_from_string(model_name)
        # get the model config class
        model_config_name = model_config_kwargs.pop("name")
        model_config_class = import_class_from_string(model_config_name)
    except AttributeError as err:
        raise err

    update_left_inplace(model_config_kwargs, kwargs)
    model_config = model_config_class(**model_config_kwargs)
    assert isinstance(model_config, PretrainedConfig)

    return model_class(model_config)


def tokenizer_from_config(
    config_path: Union[Text, Path], tokenizer_path: Union[Text, Path], **kwargs: Any
) -> Tokenizer:
    try:
        # parse config and make sure it contains tokenizer setup
        config_kwargs = parse_config(config_path)
        validate_tokenizer_config(config_kwargs)

        # extract tokenizer setup parameters
        tokenizer_kwargs = config_kwargs["tokenizer"]
        assert isinstance(tokenizer_kwargs, dict)

        # get the tokenizer class
        tokenizer_name = tokenizer_kwargs.pop("name")
        tokenizer_class = import_class_from_string(tokenizer_name)
        assert issubclass(tokenizer_class, Tokenizer)
    except AttributeError as err:
        raise err

    if isinstance(tokenizer_path, Path):
        tokenizer_path = str(tokenizer_path)

    update_left_inplace(tokenizer_kwargs, kwargs)
    tokenizer = tokenizer_class.from_pretrained(tokenizer_path, **tokenizer_kwargs)

    return tokenizer

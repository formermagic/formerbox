# coding=utf-8
# Copyright 2020 AllenNLP Authors and The FormerMagic Inc. team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
import logging
import typing
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Text, Tuple, Type

from formerbox.common.from_partial import FromPartial

RegistryType = typing.TypeVar("RegistryType", bound="Registrable")
RegistryValue = Tuple[Type["Registrable"], Optional[Text]]

logger = logging.getLogger(__name__)


class Registry:
    _registry: Dict[Text, RegistryValue]

    def __init__(self, name: Optional[Text] = None) -> None:
        self.name = name
        self._registry = {}

    def get(
        self, key: Text, value_type: Type[RegistryType]
    ) -> Tuple[Type[RegistryType], Optional[Text]]:
        cls_candidate, constructor = self._get_candidate(key)
        if issubclass(cls_candidate, value_type):
            cls = typing.cast(Type[RegistryType], cls_candidate)
            return cls, constructor
        raise RuntimeError(f"Unable to get value for key {key} of type {value_type}")

    def keys(self, value_type: Type[typing.Any]) -> List[Text]:
        result: List[Text] = []
        for key, value in self._registry.items():
            cls, _ = value
            if issubclass(cls, value_type):
                result.append(key)

        return result

    def __getitem__(self, key: Text) -> RegistryValue:
        return self._get_candidate(key)

    def __setitem__(self, key: Text, value: RegistryValue) -> None:
        self._registry[key] = value

    def __contains__(self, key: Text) -> bool:
        return key in self._registry

    def _get_candidate(self, key: Text) -> RegistryValue:
        candidate: RegistryValue
        if key in self._registry:
            candidate = self._registry[key]
        elif "." in key:
            candidate = self._get_candidate_from_module(key)
        else:
            raise RuntimeError(
                f"{key} is not a registered name for {self.name}. "
                "You probably need to use the --include-package flag "
                "to load your custom code. Alternatively, you can specify your choices "
                'using fully-qualified paths, e.g. {"model": "my_module.models.MyModel"} '
                "in which case they will be automatically imported correctly."
            )

        return candidate

    def _get_candidate_from_module(self, key: Text) -> RegistryValue:
        parts = key.split(".")
        submodule = ".".join(parts[:-1])
        class_name = parts[-1]

        try:
            module = importlib.import_module(submodule)
        except ModuleNotFoundError as err:
            raise RuntimeError(
                f"tried to interpret {key} as a path to a class "
                f"but unable to import module {submodule}"
            ) from err

        try:
            subclass = getattr(module, class_name)
            subclass = typing.cast(Type["Registrable"], subclass)
            constructor = None
        except AttributeError as err:
            raise RuntimeError(
                f"tried to interpret {key} as a path to a class "
                f"but unable to find class {class_name} in {submodule}"
            ) from err

        return (subclass, constructor)


class Registrable(FromPartial):
    _registry: Dict[Text, Registry] = defaultdict(Registry)
    _constructor: Optional[Text] = None

    @classmethod
    def register(
        cls: Type["Registrable"],
        name: Text,
        constructor: Optional[Text] = None,
        exist_ok: bool = False,
    ) -> Callable[[Type[RegistryType]], Type[RegistryType]]:
        registry = Registrable._registry[cls.__name__]
        registry.name = cls.__name__

        def add_to_registry(
            subclass: Type[RegistryType],
        ) -> Type[RegistryType]:
            if name in registry and not exist_ok:
                message = f"{name} has already been registered as {registry[name][0].__name__}"
                raise RuntimeError(message)

            registry[name] = (subclass, constructor)
            return subclass

        return add_to_registry

    @classmethod
    def resolve(
        cls: Type[RegistryType], name: Text
    ) -> Tuple[Type[RegistryType], Optional[Text]]:
        registry = cls._get_registry()
        return registry.get(name, value_type=cls)

    @classmethod
    def from_registry(
        cls: Type[RegistryType], name: Text
    ) -> Tuple[Type[RegistryType], Callable[..., RegistryType]]:
        registry = cls._get_registry()
        subclass, constructor_name = registry.get(name, value_type=cls)
        if constructor_name is None:
            constructor = typing.cast(Callable, subclass)
        else:
            constructor = getattr(subclass, constructor_name)
            constructor = typing.cast(Callable, constructor)

        return subclass, constructor

    @classmethod
    def from_name(cls: Type[RegistryType], name: Text) -> Callable[..., RegistryType]:
        registry = cls._get_registry()
        subclass, constructor_name = registry.get(name, value_type=cls)
        if constructor_name is None:
            constructor = typing.cast(Callable, subclass)
        else:
            constructor = getattr(subclass, constructor_name)
            constructor = typing.cast(Callable, constructor)

        return constructor

    @classmethod
    def list_available(cls) -> List[Text]:
        registry = cls._get_registry()
        registered_names = registry.keys(cls)

        result: List[Text]
        if cls._constructor is None:
            result = registered_names
        elif cls._constructor not in registered_names:
            raise RuntimeError(
                f"Default implementation {cls._constructor} is not registered"
            )
        else:
            result = [cls._constructor]
            result += [name for name in registered_names if name != cls._constructor]

        return result

    @classmethod
    def _get_registry(cls: Type["Registrable"]) -> Registry:
        if cls.__name__ not in Registrable._registry:
            raise RuntimeError(f"{cls.__name__} is not a registered name")
        return Registrable._registry[cls.__name__]

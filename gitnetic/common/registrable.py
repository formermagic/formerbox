# coding=utf-8
# Copyright 2020 AllenNLP Authors and The Former Magic Inc. team.
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
from collections import defaultdict
from typing import Callable, Dict, Optional, Text, Tuple, Type, TypeVar, Union

from gitnetic.common.from_args import FromArgs

T = TypeVar("T", bound="Registrable")

RegistryKey = Union[Text, Type[T]]
RegistryRecord = Tuple[Type[T], Optional[Text]]
RegistryMap = Dict[Text, RegistryRecord]


class Registrable(FromArgs):
    _registry: Dict[RegistryKey, RegistryMap] = defaultdict(dict)
    default_implementation: Optional[Text] = None

    @classmethod
    def register(
        cls: Type[T],
        name: Text,
        constructor: Optional[Text] = None,
        exist_ok: bool = False,
    ) -> Callable[[Type[T]], Type[T]]:
        registry = Registrable._registry[cls]

        def add_subclass_to_registry(subclass: Type[T]) -> Type[T]:
            # Add to registry, raise an error if key has already been used.
            if name in registry:
                if not exist_ok:
                    message = (
                        f"{name} has already been registered as {registry[name][0].__name__}, but "
                        f"exist_ok=True, so overwriting with {cls.__name__}"
                    )
                    raise RuntimeError(message)

            registry[name] = (subclass, constructor)
            return subclass

        return add_subclass_to_registry

    @classmethod
    def resolve_class_name(cls: Type[T], name: Text) -> RegistryRecord:
        subclass: Type[T]
        constructor: Optional[Text]

        if name in Registrable._registry[cls]:
            subclass, constructor = Registrable._registry[cls][name]
        elif "." in name:
            # This might be a fully qualified class name, so we'll try importing its "module"
            # and finding it there.
            parts = name.split(".")
            submodule = ".".join(parts[:-1])
            class_name = parts[-1]

            try:
                module = importlib.import_module(submodule)
            except ModuleNotFoundError as err:
                raise RuntimeError(
                    f"tried to interpret {name} as a path to a class "
                    f"but unable to import module {submodule}"
                ) from err

            try:
                subclass = getattr(module, class_name)
                constructor = None
            except AttributeError as err:
                raise RuntimeError(
                    f"tried to interpret {name} as a path to a class "
                    f"but unable to find class {class_name} in {submodule}"
                ) from err

        else:
            # is not a qualified class name
            raise RuntimeError(
                f"{name} is not a registered name for {cls.__name__}. "
                "You probably need to use the --include-package flag "
                "to load your custom code. Alternatively, you can specify your choices "
                """using fully-qualified paths, e.g. {"model": "my_module.models.MyModel"} """
                "in which case they will be automatically imported correctly."
            )

        return subclass, constructor

    @classmethod
    def from_registry(cls: Type[T], name: Text) -> Callable[..., T]:
        subclass, constructor = cls.resolve_class_name(name)
        if constructor is None:
            return subclass
        return getattr(subclass, constructor)
import importlib
from collections import defaultdict
from typing import Callable, Dict, Optional, Text, Type, TypeVar, Union

from gitnetic.common.from_args import FromArgs

T = TypeVar("T", bound="Registrable")


class Registrable(FromArgs):
    _registry: Dict[Union[Text, Type], Dict[Text, Type]] = defaultdict(dict)
    default_implementation: Optional[Text] = None

    @classmethod
    def register(
        cls: Type[T], name: Text, exist_ok: bool = False
    ) -> Callable[[Type[T]], Type[T]]:
        registry = Registrable._registry[cls]

        def add_subclass_to_registry(subclass: Type[T]) -> Type[T]:
            # Add to registry, raise an error if key has already been used.
            if name in registry:
                if not exist_ok:
                    message = (
                        f"{name} has already been registered as {registry[name].__name__}, but "
                        f"exist_ok=True, so overwriting with {cls.__name__}"
                    )
                    raise RuntimeError(message)

            registry[name] = subclass
            return subclass

        return add_subclass_to_registry

    @classmethod
    def resolve_class_name(cls: Type[T], name: Text) -> Type[T]:
        subclass: Type[T]
        if name in Registrable._registry[cls]:
            subclass = Registrable._registry[cls][name]
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

        return subclass

    @classmethod
    def from_registry(cls: Type[T], name: Text) -> Callable[..., T]:
        subclass = cls.resolve_class_name(name)
        return subclass

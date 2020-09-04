import inspect
from typing import Any, Callable, Dict, Text, Type, TypeVar

T = TypeVar("T")


class FromArgs:
    @classmethod
    def from_args(cls: Type[T], args: Dict[Text, Any], **kwargs: Any) -> T:
        valid_kwargs = inspect.signature(cls.__init__).parameters
        obj_kwargs = dict((name, args[name]) for name in valid_kwargs if name in args)
        obj_kwargs.update(**kwargs)
        return cls(**obj_kwargs)

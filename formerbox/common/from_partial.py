import inspect
import typing
from typing import Any, Callable, Type, TypeVar

T = TypeVar("T")  # pylint: disable=invalid-name


class FromPartial:
    @classmethod
    def from_partial(cls: Type[T], **kwargs: Any) -> T:
        # inspect the instance init method signature
        signature = inspect.signature(cls.__init__)
        # select only excplicit instance attributes
        obj_kwargs = {}
        for key in signature.parameters:
            try:
                obj_kwargs[key] = kwargs[key]
            except KeyError:
                continue

        # build an instance with selected attributes
        assert inspect.isclass(cls)
        constructor = typing.cast(Callable[..., T], cls)
        initialized = constructor(**obj_kwargs)
        return initialized

import inspect
from typing import Any, Type, TypeVar

T = TypeVar("T")  # pylint: disable=invalid-name


class PartialInitable:
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
        return cls(**obj_kwargs)

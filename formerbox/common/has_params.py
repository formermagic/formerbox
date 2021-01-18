import typing
from typing import Any, Type

# pylint: disable=unused-import
from formerbox.common.dataclass_argparse import DataclassArgumentParser, DataclassBase
from formerbox.common.from_partial import FromPartial
from typing_extensions import Protocol

Self = typing.TypeVar("Self", bound="FromPartial")
ParamsType = typing.TypeVar("ParamsType", bound=DataclassBase)


class HasParams(Protocol[ParamsType]):
    params: ParamsType

    @classmethod
    def params_type(cls) -> Type[ParamsType]:
        raise NotImplementedError()

    @classmethod
    def from_params(cls: Type[Self], params: ParamsType, **kwargs: Any) -> Self:
        obj = cls.from_partial(params=params, **kwargs)
        obj = typing.cast(Self, obj)
        return obj


class HasParsableParams(HasParams[ParamsType]):
    @classmethod
    def add_argparse_params(cls, parser: DataclassArgumentParser) -> None:
        params_type = typing.cast(Type[DataclassBase], cls.params_type)
        parser.add_arguments(params_type)

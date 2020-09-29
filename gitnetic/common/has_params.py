import typing
from typing import Any, Type

# pylint: disable=unused-import
from gitnetic.common.dataclass_argparse import DataclassArgumentParser, DataclassBase
from gitnetic.common.partial_initable import PartialInitable
from typing_extensions import Protocol

Self = typing.TypeVar("Self", bound="PartialInitable")
ParamsType = typing.TypeVar("ParamsType", covariant=True, bound=DataclassBase)


class HasParams(Protocol[ParamsType]):
    params: ParamsType

    @property
    @classmethod
    def params_type(cls) -> Type[ParamsType]:
        raise NotImplementedError()

    @classmethod
    def from_params(cls: Type[Self], params: ParamsType, **kwargs: Any) -> Self:
        return cls.from_partial(params=params, **kwargs)


class HasParsableParams(HasParams[ParamsType]):
    @classmethod
    def add_argparse_params(cls, parser: DataclassArgumentParser) -> None:
        parser.add_arguments(cls.params_type)  # type: ignore
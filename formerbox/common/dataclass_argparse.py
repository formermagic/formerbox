import collections
import dataclasses
import logging
import sys
import typing
from argparse import ArgumentParser, Namespace
from dataclasses import Field, dataclass
from enum import Enum
from inspect import isclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, List, Optional, Text, Tuple, Type, Union

from formerbox.common.registrable import Registrable
from formerbox.utils import str2bool
from typing_inspect import (
    get_args,
    get_origin,
    is_literal_type,
    is_optional_type,
    is_tuple_type,
    is_union_type,
)

MISSING: Any = "???"

ParamsType = typing.TypeVar("ParamsType")
DataclassBaseType = Type["DataclassBase"]
DataclassTypes = Union[DataclassBaseType, typing.Iterable[DataclassBaseType]]

logger = logging.getLogger(__name__)


def is_missing(obj: Any) -> bool:
    return obj is dataclasses.MISSING or obj == MISSING


def get_parsed_attr(
    params: Tuple[Union["DataclassBase", Namespace], ...],
    attribute_name: Text,
    default: Optional[Any] = None,
) -> Optional[Any]:
    attribute = default
    for args in params:
        if hasattr(args, attribute_name):
            attribute = getattr(args, attribute_name)
    return attribute


def get_params_item(
    params: Tuple[Union["DataclassBase", Namespace], ...],
    params_type: Type[ParamsType],
    default: Optional[Any] = None,
) -> Optional[ParamsType]:
    for args in params:
        if isinstance(args, params_type):
            return typing.cast(ParamsType, args)
    return default


@dataclass
class DataclassBase:
    @classmethod
    def fields(cls) -> Dict[Text, Field]:
        # pylint: disable=no-member
        assert hasattr(cls, "__dataclass_fields__"), (
            "You don't seem to wrap around @dataclass decorator."
            " Please make sure your class wraps around @dataclass."
        )

        fields = cls.__dataclass_fields__
        return fields

    @classmethod
    def attributes(cls) -> List[Text]:
        return [name for name, _ in cls.fields().items()]

    @classmethod
    def get_field(cls, attribute_name: Text) -> Field:
        try:
            return cls.fields()[attribute_name]
        except KeyError as error:
            raise KeyError(
                f"Unable to find an attribute for name '{attribute_name}'."
                " You might have forgotten to wrap your class around @dataclass"
                " decorator, so we're unable to find the requested attribute."
            ) from error

    @classmethod
    def get_metadata(
        cls,
        attribute_name: Text,
        metadata_key: Optional[Text] = None,
        metadata_default_value: Optional[Any] = None,
    ) -> Optional[Any]:
        metadata = cls.get_field(attribute_name).metadata
        if metadata_key is not None:
            return metadata.get(metadata_key, metadata_default_value)
        return metadata

    @classmethod
    def get_type(cls, attribute_name: Text) -> Type[Any]:
        return cls.get_field(attribute_name).type

    @classmethod
    def get_help(cls, attribute_name: Text) -> Optional[Text]:
        return cls.get_metadata(attribute_name, metadata_key="help")

    @classmethod
    def get_choices(cls, attribute_name: Text) -> Optional[List[Text]]:
        return cls.get_metadata(attribute_name, metadata_key="choices")


def is_enum_type(dtype: Type[Any]) -> bool:
    if not isinstance(dtype, type):
        return False
    return issubclass(dtype, Enum)


def is_list_type(dtype: Type[Any]) -> bool:
    origin = get_origin(dtype)
    if origin is None:
        return False
    return issubclass(origin, list)


class DataclassArgumentParser(ArgumentParser):
    def __init__(
        self,
        dataclass_types: Optional[DataclassTypes] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        # add dataclass args if specified
        if dataclass_types is not None:
            # make sure the `dataclass_types` is a list
            if not isinstance(dataclass_types, collections.Iterable):
                self.dataclass_types = [dataclass_types]
            else:
                self.dataclass_types = list(dataclass_types)

            # add dataclass fields as argparser arguments
            for dataclass_type in self.dataclass_types:
                self.add_arguments(dataclass_type)
        else:
            self.dataclass_types = []

    def add_arguments(self, dataclass_type: DataclassBaseType) -> None:
        if dataclass_type not in self.dataclass_types:
            self.dataclass_types.append(dataclass_type)

        for attribute in dataclass_type.attributes():
            arg_name = f"--{attribute}"

            field_obj = dataclass_type.get_field(attribute)
            field_type = dataclass_type.get_type(attribute)
            metadata = dataclass_type.get_metadata(attribute)

            # prepare a dict from the metadata attribute
            kwargs: Dict[Text, Any] = {}
            if isinstance(metadata, MappingProxyType):
                kwargs = dict(metadata)
            elif isinstance(metadata, dict):
                kwargs = metadata
            else:
                assert False, f"Unsupported metadata property type {type(metadata)}."

            # list available choices via registrable objects
            try:
                choices = kwargs["choices"]
                if isclass(choices) and issubclass(choices, Registrable):
                    kwargs["choices"] = sorted(choices.list_available())
            except KeyError:
                pass

            # parse field primitive types if possible
            for primitive_type in (int, float, str):
                if not is_optional_type(field_type):
                    continue

                # get container generic type
                container_type = get_args(field_type)[0]
                container_type = typing.cast(Type[Any], container_type)

                for collection_type in (List,):
                    if container_type == collection_type[primitive_type]:
                        field_type = collection_type[primitive_type]
                if container_type == primitive_type:
                    field_type = primitive_type

            # unpack optional container
            if is_optional_type(field_type):
                optional_type = get_args(field_type)[0]
                field_type = typing.cast(Type[Any], optional_type)

            # raise value errors for unsupported field types
            if is_union_type(field_type):
                raise TypeError("Unsupported argument type")
            if is_tuple_type(field_type):
                raise TypeError("Unsupported argument type")

            # parse supported fields
            if is_enum_type(field_type):
                choices = typing.cast(Type[Enum], field_type)
                kwargs["choices"] = list(choices)
                kwargs["type"] = field_type
                if not is_missing(field_obj.default):
                    kwargs["default"] = field_obj.default
                else:
                    kwargs["required"] = True
            elif is_literal_type(field_type):
                choices = get_args(field_type)
                field_types = set(type(x) for x in choices)
                assert (
                    len(field_types) == 1
                ), f"{attribute} cannot be a choice of mixed types"

                kwargs["choices"] = choices
                kwargs["type"] = field_types.pop()
                if not is_missing(field_obj.default):
                    kwargs["default"] = field_obj.default
                else:
                    kwargs["required"] = True
            elif field_type is bool:
                kwargs["type"] = str2bool
                if not is_missing(field_obj.default):
                    kwargs["default"] = field_obj.default
                else:
                    kwargs["required"] = True
            elif is_list_type(field_type):
                container_types = get_args(field_type)
                assert (
                    len(set(container_types)) == 1
                ), f"{attribute} cannot be a list of mixed types"

                kwargs["nargs"] = "+"
                kwargs["type"] = container_types[0]
                if not is_missing(field_obj.default):
                    kwargs["default"] = field_obj.default
                elif not is_missing(field_obj.default_factory):
                    kwargs["default"] = field_obj.default_factory()
                else:
                    kwargs["required"] = True
            else:
                kwargs["type"] = field_type
                if not is_missing(field_obj.default):
                    kwargs["default"] = field_obj.default
                elif not is_missing(field_obj.default_factory):
                    kwargs["default"] = field_obj.default_factory()
                else:
                    kwargs["required"] = True

            self.add_argument(arg_name, **kwargs)

    def parse_args_into_dataclasses(
        self,
        args: Optional[typing.Sequence[Text]] = None,
        return_remaining_strings: bool = False,
        look_for_args_file: bool = True,
    ) -> Tuple[Union[DataclassBase, Namespace], ...]:
        if look_for_args_file and sys.argv:
            args_file = Path(sys.argv[0]).with_suffix(".args")
            if args_file.exists():
                fargs = args_file.read_text().split()
                args = fargs + list(args) if args is not None else fargs + sys.argv[1:]
                # in case of duplicate arguments the first one has precedence
                # so we append rather than prepend.

        namespace, remaining_args = self.parse_known_args(args=args)
        outputs = []
        for dtype in self.dataclass_types:
            keys = set(dtype.attributes())
            inputs = {k: v for k, v in vars(namespace).items() if k in keys}
            for k in keys:
                delattr(namespace, k)
            obj = dtype(**inputs)
            outputs.append(obj)

        if len(vars(namespace)) > 0:
            # additional namespace
            outputs.append(namespace)

        result: Tuple[Union[DataclassBase, Namespace], ...]
        if return_remaining_strings:
            result = (*outputs, remaining_args)
        else:
            if remaining_args:
                raise ValueError(
                    f"Some specified arguments are not used by the parser: {remaining_args}"
                )

            result = (*outputs,)
        return result

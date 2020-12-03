import logging
from abc import abstractmethod
from argparse import _SubParsersAction
from typing import Any, Callable, Dict, Optional, Text, Tuple, Type, TypeVar

from formerbox.common.dataclass_argparse import DataclassArgumentParser
from formerbox.common.registrable import Registrable

T = TypeVar("T", bound="Subcommand")  # pylint: disable=invalid-name

logger = logging.getLogger(__name__)


class Subcommand(Registrable):
    _reverse_registry: Dict[Type["Subcommand"], Text] = {}

    @property
    def name(self) -> Text:
        return self._reverse_registry[self.__class__]

    @abstractmethod
    def add_subparser(
        self, parser: _SubParsersAction
    ) -> Tuple[DataclassArgumentParser, Dict[Text, Any]]:
        raise NotImplementedError()

    @classmethod
    def register(
        cls,
        name: Text,
        constructor: Optional[Text] = None,
        exist_ok: bool = False,
    ) -> Callable[[Type[T]], Type[T]]:
        super_register_fn = super().register(
            name, constructor=constructor, exist_ok=exist_ok
        )

        def add_name_to_reverse_registry(subclass: Type[T]) -> Type[T]:
            subclass = super_register_fn(subclass)
            # Don't need to check `exist_ok`, as it's done by super.
            # Also, don't need to delete previous entries if overridden, they can just stay there.
            cls._reverse_registry[subclass] = name
            return subclass

        return add_name_to_reverse_registry

    def _add_parser(
        self, parser: _SubParsersAction, **kwargs: Any
    ) -> DataclassArgumentParser:
        subparser = parser.add_parser(**kwargs)
        assert isinstance(subparser, DataclassArgumentParser)
        return subparser

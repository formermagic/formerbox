from abc import ABCMeta, abstractmethod
from argparse import Namespace
from typing import Tuple, Type, TypeVar, Union

from formerbox.common.dataclass_argparse import DataclassBase
from formerbox.common.has_params import HasParsableParams
from formerbox.common.registrable import Registrable
from formerbox.modules import TransformerDataModule, TransformerModule
from transformers import PreTrainedTokenizerBase
from typing_extensions import Protocol

Tokenizer = PreTrainedTokenizerBase

ModuleType = TypeVar("ModuleType", bound=TransformerModule)
DataModuleType = TypeVar("DataModuleType", bound=TransformerDataModule)
TaskModuleType = TypeVar("TaskModuleType", bound="TaskModule")

ParamType = Union[DataclassBase, Namespace]


class TaskModuleBase(Protocol[ModuleType, DataModuleType]):
    ...


class TaskModule(
    TaskModuleBase[ModuleType, DataModuleType],
    Registrable,
    HasParsableParams,
    metaclass=ABCMeta,
):
    def __init__(
        self,
        tokenizer: Tokenizer,
        module: ModuleType,
        datamodule: DataModuleType,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.module = module
        self.datamodule = datamodule

    @classmethod
    @abstractmethod
    def setup(
        cls: Type[TaskModuleType], params: Tuple[ParamType, ...]
    ) -> TaskModuleType:
        raise NotImplementedError()

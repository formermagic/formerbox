from abc import ABCMeta, abstractmethod
from argparse import Namespace
from typing import Tuple, Type, TypeVar, Union

from formerbox.common.dataclass_argparse import DataclassBase
from formerbox.common.has_params import HasParsableParams
from formerbox.common.registrable import Registrable

# pylint: disable=unused-import
from formerbox.modules import TransformerDataModule, TransformerModule
from transformers import PreTrainedTokenizerBase
from typing_extensions import Protocol

T = TypeVar("T", bound="TaskModule")  # pylint: disable=invalid-name
Tokenizer = PreTrainedTokenizerBase
ModuleType = TypeVar("ModuleType", bound="TransformerModule")
DataModuleType = TypeVar("DataModuleType", bound="TransformerDataModule")
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
    def setup(cls: Type[T], params: Tuple[ParamType, ...]) -> T:
        raise NotImplementedError()

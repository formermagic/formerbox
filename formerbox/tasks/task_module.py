import logging
from abc import abstractmethod
from typing import Tuple, Type, TypeVar

from formerbox.common.has_params import HasParsableParams, ParamsType
from formerbox.common.registrable import Registrable
from transformers import PreTrainedTokenizerFast as Tokenizer
from typing_extensions import Protocol

ModuleType = TypeVar("ModuleType")
DataModuleType = TypeVar("DataModuleType")
TaskModuleType = TypeVar("TaskModuleType", bound="TaskModule")

logger = logging.getLogger(__name__)


class TaskModuleBase(Protocol[ModuleType, DataModuleType]):
    ...


class TaskModule(
    TaskModuleBase[ModuleType, DataModuleType],
    HasParsableParams[ParamsType],
    Registrable,
):
    params: ParamsType
    params_type: Type[ParamsType]

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
        cls: Type[TaskModuleType], params: Tuple[ParamsType, ...]
    ) -> TaskModuleType:
        raise NotImplementedError()

import logging
from abc import ABCMeta, abstractmethod
from argparse import Namespace
from typing import Tuple, Type, TypeVar, Union

from formerbox.common.dataclass_argparse import DataclassBase, get_params_item
from formerbox.common.has_params import HasParsableParams, ParamsType
from formerbox.common.registrable import Registrable
from pytorch_lightning import LightningDataModule, LightningModule
from transformers import PreTrainedTokenizerFast as Tokenizer

ModuleType = TypeVar("ModuleType", bound=LightningModule)
DataModuleType = TypeVar("DataModuleType", bound=LightningDataModule)
TaskModuleType = TypeVar("TaskModuleType", bound="TaskModule")

logger = logging.getLogger(__name__)


class TaskModule(
    HasParsableParams[ParamsType],
    Registrable,
    metaclass=ABCMeta,
):
    params: ParamsType
    params_type: Type[ParamsType]

    @abstractmethod
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
        cls: Type[TaskModuleType],
        params: Tuple[Union[DataclassBase, Namespace], ...],
    ) -> TaskModuleType:
        raise NotImplementedError()

import logging
from abc import ABCMeta, abstractmethod
from argparse import Namespace
from typing import Tuple, Type, TypeVar, Union

from formerbox.common.dataclass_argparse import DataclassBase, get_params_item
from formerbox.common.has_params import HasParsableParams, ParamsType
from formerbox.common.registrable import Registrable
from pytorch_lightning import LightningDataModule, LightningModule
from transformers import PreTrainedTokenizerFast as Tokenizer

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
        module: LightningModule,
        datamodule: LightningDataModule,
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

    @classmethod
    def get_params(
        cls,
        params: Tuple[Union[DataclassBase, Namespace], ...],
        params_type: Type[ParamsType],
    ) -> ParamsType:
        item = get_params_item(
            params=params,
            params_type=params_type,
        )

        assert item is not None, (
            f"Unable to find params of type ({params_type}) in {params}.",
            " Make sure all required methods are implemented in a running task.",
            " You may also want to make sure all required arguments are specified.",
        )

        return item

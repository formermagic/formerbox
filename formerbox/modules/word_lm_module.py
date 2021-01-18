import logging
from dataclasses import dataclass
from typing import Type

from formerbox.modules.translation_module import TranslationModule
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizerFast as Tokenizer

logger = logging.getLogger(__name__)


class WordLMModule(TranslationModule):
    @dataclass
    class Params(TranslationModule.Params):
        pass

    params: Params
    params_type: Type[Params] = Params

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: Tokenizer,
        params: Params,
    ) -> None:
        super().__init__(model, tokenizer, params)

        # save the arguments to easily restore
        # from the saved pytorch checkpoint
        self.save_hyperparameters()

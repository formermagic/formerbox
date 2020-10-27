import logging
from typing import Any, List, Optional, Text, Union

from formerbox.tasks import ByteLevelBPETokenizerFast
from formerbox.utils.code_tokenizer import SpecialToken
from tokenizers import AddedToken
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

Token = Union[Text, AddedToken]
TransformersTokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

logger = logging.getLogger(__name__)


VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {},
    "merges_file": {},
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {}


class CodeBBPETokenizerFast(ByteLevelBPETokenizerFast):
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["attention_mask"]

    def __init__(
        self,
        vocab_file: Text,
        merges_file: Text,
        tokenizer_file: Optional[Text] = None,
        errors: Text = "replace",
        unk_token: Token = "<unk>",
        bos_token: Token = "<s>",
        eos_token: Token = "</s>",
        pad_token: Token = "<pad>",
        sep_token: Token = "</s>",
        cls_token: Token = "<s>",
        mask_token: Token = "<mask>",
        add_prefix_space: bool = False,
        trim_offsets: bool = True,
        lowercase: bool = False,
        dropout: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        # pylint: disable=too-many-locals, too-many-arguments
        super().__init__(
            vocab_file,
            merges_file,
            tokenizer_file=tokenizer_file,
            errors=errors,
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            add_prefix_space=add_prefix_space,
            trim_offsets=trim_offsets,
            lowercase=lowercase,
            dropout=dropout,
            **kwargs,
        )

        # add code specific special tokens
        additional_special_tokens: List[Text] = []
        for token in SpecialToken:
            setattr(self, token.name, token.value)
            additional_special_tokens.append(token.value)

        self.additional_special_tokens += additional_special_tokens

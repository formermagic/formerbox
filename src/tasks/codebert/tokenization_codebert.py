import logging
from typing import Any, List, Optional, Text

from tokenizers.normalizers import Lowercase
from transformers import RobertaTokenizerFast

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

# pylint: disable=abstract-method
class CodeBertTokenizerFast(RobertaTokenizerFast):

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        vocab_file: Text,
        merges_file: Text,
        bos_token: Text = "<s>",
        eos_token: Text = "</s>",
        sep_token: Text = "</s>",
        cls_token: Text = "<s>",
        unk_token: Text = "<unk>",
        pad_token: Text = "<pad>",
        mask_token: Text = "<mask>",
        newline_token: Text = "<nl>",
        add_prefix_space: bool = True,
        trim_offsets: bool = True,
        lowercase: bool = True,
        **kwargs: Any,
    ) -> None:
        # proxy original roberta fast tokenizer init
        super().__init__(
            vocab_file=vocab_file,
            merges_file=merges_file,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            add_prefix_space=add_prefix_space,
            trim_offsets=trim_offsets,
            additional_special_tokens=[newline_token],
            kwargs=kwargs,
        )

        # a special token for new lines in text datasets
        self.newline_token = newline_token
        self.newline_token_id = self.backend_tokenizer.token_to_id(newline_token)

        # validate if token ids are in the correct order
        assert self.pad_token_id == 1, "`pad_token_id` must always be at index 1"

        # add lowercase normalizer if needed
        if lowercase:
            self.backend_tokenizer._tokenizer.normalizer = Lowercase()

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formated with special tokens for the model."
                )

            def decode_idx(value: int) -> int:
                if value in [self.sep_token_id, self.cls_token_id]:
                    return 1
                return 0

            return [decode_idx(idx) for idx in token_ids_0]

        first_seq = [1] + ([0] * len(token_ids_0)) + [1]
        if token_ids_1 is None:
            return first_seq

        second_seq = [1] + ([0] * len(token_ids_1)) + [1]
        return first_seq + second_seq

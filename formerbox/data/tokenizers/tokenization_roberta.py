import logging
from typing import Any, Optional, Text, Union

from formerbox.data.tokenizers.tokenization_base import TokenizerBase
from transformers import RobertaTokenizerFast

from tokenizers import AddedToken, pre_tokenizers, processors
from tokenizers.normalizers import Lowercase, Sequence

Token = Union[Text, AddedToken]

logger = logging.getLogger(__name__)


@TokenizerBase.register("roberta")
class RobertaTokenizer(RobertaTokenizerFast, TokenizerBase):
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
        **kwargs: Any,
    ) -> None:
        # pylint: disable=too-many-arguments, too-many-locals
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
            **kwargs,
        )

        # setup tokenizer normalization
        normalizer = Lowercase() if lowercase else Sequence([])
        self.backend_tokenizer.normalizer = normalizer
        self.lowercase = lowercase

        # setup tokenizer pre-tokenization
        pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=add_prefix_space)
        self.backend_tokenizer.pre_tokenizer = pre_tokenizer
        self.add_prefix_space = add_prefix_space

        # setup tokenizer post-processing
        assert self.sep_token_id is not None
        assert self.cls_token_id is not None

        post_processor = processors.RobertaProcessing(
            sep=(self.sep_token, self.sep_token_id),
            cls=(self.cls_token, self.cls_token_id),
            trim_offsets=trim_offsets,
            add_prefix_space=add_prefix_space,
        )

        self.backend_tokenizer.post_processor = post_processor
        self.trim_offsets = trim_offsets

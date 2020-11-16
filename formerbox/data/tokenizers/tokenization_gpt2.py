import logging
from typing import Any, Optional, Text, Union

from formerbox.data.tokenizers.tokenization_base import TokenizerBase
from transformers import GPT2TokenizerFast

from tokenizers import AddedToken, pre_tokenizers, processors
from tokenizers.normalizers import Lowercase, Sequence

Token = Union[Text, AddedToken]

logger = logging.getLogger(__name__)


@TokenizerBase.register("gpt2")
class GPT2Tokenizer(GPT2TokenizerFast, TokenizerBase):
    def __init__(
        self,
        vocab_file: Text,
        merges_file: Text,
        tokenizer_file: Optional[Text] = None,
        unk_token: Token = "<|endoftext|>",
        bos_token: Token = "<|endoftext|>",
        eos_token: Token = "<|endoftext|>",
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
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
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
        post_processor = processors.ByteLevel(trim_offsets=trim_offsets)
        self.backend_tokenizer.post_processor = post_processor
        self.trim_offsets = trim_offsets

import logging
from typing import Any, List, Optional, Text, Union

from formerbox.data.tokenizers.tokenization_base import (
    Padding,
    Seq2SeqTokenizer,
    TokenizerBase,
    Truncation,
)
from transformers import BartTokenizerFast, BatchEncoding, TensorType

from tokenizers import AddedToken, pre_tokenizers, processors
from tokenizers.normalizers import Lowercase, Sequence

Token = Union[Text, AddedToken]

logger = logging.getLogger(__name__)


VERY_LARGE_INTEGER = int(1e30)

# pylint: disable=arguments-differ
@TokenizerBase.register("bart")
class BartTokenizer(BartTokenizerFast, Seq2SeqTokenizer, TokenizerBase):
    prefix_tokens: List[int] = []

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

    def prepare_seq2seq_batch(
        self,
        src_texts: List[Text],
        tgt_texts: Optional[List[Text]] = None,
        max_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
        padding: Union[Padding, bool] = "longest",
        truncation: Union[Truncation, bool] = True,
        return_tensors: Optional[TensorType] = None,
        **kwargs,
    ) -> BatchEncoding:
        # pylint: disable=too-many-arguments
        # make sure the max length is set
        if max_length is None:
            max_length = getattr(self, "model_max_length", VERY_LARGE_INTEGER)
        if max_target_length is None:
            max_target_length = max_length

        # process the source texts
        self.prefix_tokens = []
        model_inputs = self(
            src_texts,
            add_special_tokens=True,
            return_tensors=return_tensors,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            **kwargs,
        )

        if tgt_texts is None:
            return model_inputs

        # process the target texts
        assert self.pad_token_id is not None
        self.prefix_tokens = [self.pad_token_id]
        labels_and_decoder_mask = self(
            tgt_texts,
            add_special_tokens=True,
            return_tensors=return_tensors,
            padding=padding,
            max_length=max_target_length,
            truncation=truncation,
            **kwargs,
        )

        # target labels
        model_inputs["labels"] = labels_and_decoder_mask["input_ids"]

        # clear the prefix tokens
        self.prefix_tokens = []

        return model_inputs

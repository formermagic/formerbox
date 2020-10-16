from typing import Any, List, Optional, Text, Union

from tokenizers import AddedToken
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import RobertaProcessing
from transformers import BatchEncoding, PreTrainedTokenizer, PreTrainedTokenizerFast

Token = Union[Text, AddedToken]
TransformersTokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]


VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {},
    "merges_file": {},
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {}


class TransformerTokenizerFast(PreTrainedTokenizerFast):
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["attention_mask"]

    def __init__(
        self,
        vocab_file: Text,
        merges_file: Text,
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
        # pylint: disable=too-many-arguments, too-many-locals
        # Mask token behave like a normal word, i.e. include the space before it
        if isinstance(mask_token, str):
            mask_token = AddedToken(mask_token, lstrip=True, rstrip=False)

        kwargs.setdefault("pad_token", pad_token)
        kwargs.setdefault("sep_token", sep_token)
        kwargs.setdefault("cls_token", cls_token)
        kwargs.setdefault("mask_token", mask_token)

        super().__init__(
            ByteLevelBPETokenizer(
                vocab_file=vocab_file,
                merges_file=merges_file,
                add_prefix_space=add_prefix_space,
                trim_offsets=trim_offsets,
                lowercase=lowercase,
                dropout=dropout,
            ),
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            **kwargs,
        )

        self.add_prefix_space = add_prefix_space

        # This will add the necessary special tokens to the vocabulary if needed
        self.sanitize_special_tokens()

        assert self.sep_token_id is not None, "`sep_token_id` should not be none"
        assert self.cls_token_id is not None, "`cls_token_id` should not be none"
        self.backend_tokenizer._tokenizer.post_processor = RobertaProcessing(
            sep=(str(sep_token), self.sep_token_id),
            cls=(str(cls_token), self.cls_token_id),
            add_prefix_space=add_prefix_space,
            trim_offsets=trim_offsets,
        )

    def _batch_encode_plus(self, *args: Any, **kwargs: Any) -> BatchEncoding:
        # pylint: disable=signature-differs
        is_pretokenized = kwargs.get("is_pretokenized", False)
        assert self.add_prefix_space or not is_pretokenized, (
            f"You need to instantiate {self.__class__.__name__} with add_prefix_space=True "
            "to use it with pretokenized inputs."
        )

        return super()._batch_encode_plus(*args, **kwargs)

    def _encode_plus(self, *args: Any, **kwargs: Any) -> BatchEncoding:
        # pylint: disable=signature-differs
        is_pretokenized = kwargs.get("is_pretokenized", False)
        assert self.add_prefix_space or not is_pretokenized, (
            f"You need to instantiate {self.__class__.__name__} with add_prefix_space=True "
            "to use it with pretokenized inputs."
        )

        return super()._encode_plus(*args, **kwargs)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        assert self.bos_token_id is not None, "`bos_token_id` should not be none"
        assert self.eos_token_id is not None, "`eos_token_id` should not be none"
        output = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
        if token_ids_1 is None:
            return output

        return output + [self.eos_token_id] + token_ids_1 + [self.eos_token_id]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Creates a mask from the two sequences passed to be used in a sequence-pair classification task.
        RoBERTa does not make use of token type ids, therefore a list of zeros is returned.

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of ids.
            token_ids_1 (:obj:`List[int]`, `optional`, defaults to :obj:`None`):
                Optional second list of IDs for sequence pairs.

        Returns:
            :obj:`List[int]`: List of zeros.

        """
        assert self.sep_token_id is not None, "`sep_token_id` should not be none"
        assert self.cls_token_id is not None, "`cls_token_id` should not be none"
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]

from abc import abstractmethod
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, List, Optional, Text, Union

from tokenizers import AddedToken
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import RobertaProcessing
from transformers import (
    BatchEncoding,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    RobertaTokenizerFast,
)

from gitnetic.common.registrable import Registrable
from gitnetic.utils.utils import path_to_posix

Token = Union[Text, AddedToken]
TransformersTokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]


def fix_tokenizer(tokenizer: TransformersTokenizer) -> None:
    init_kwargs = {}
    for key, value in tokenizer.init_kwargs.items():
        if isinstance(value, AddedToken):
            init_kwargs[key] = str(value)
        else:
            init_kwargs[key] = value

    tokenizer.init_kwargs = init_kwargs


class TokenizerTrainer(Registrable):
    @abstractmethod
    def train(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError()

    @abstractmethod
    def save_pretrained(self, tokenizer_output_path: Union[Text, Path]) -> None:
        raise NotImplementedError()

    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # fmt: off
        parser.add_argument("--tokenizer_trainer_name", type=str, default=None, required=True,
                            help="")
        # fmt: on
        return parser


@TokenizerTrainer.register(
    name="transformer-tokenizer-trainer", constructor="from_args"
)
class TransformerTokenizerTrainer(TokenizerTrainer):
    def __init__(
        self,
        add_prefix_space: bool = False,
        lowercase: bool = False,
        dropout: Optional[float] = None,
        unicode_normalizer: Optional[Text] = None,
        continuing_subword_prefix: Optional[Text] = None,
        end_of_word_suffix: Optional[Text] = None,
        trim_offsets: bool = False,
    ) -> None:
        # pylint: disable=too-many-arguments
        self.add_prefix_space = add_prefix_space
        self.lowercase = lowercase
        self.dropout = dropout
        self.trim_offsets = trim_offsets
        self.special_tokens: List[Token] = [
            "<s>",
            "<pad>",
            "</s>",
            "<unk>",
            "<mask>",
        ]

        self.tokenizer = ByteLevelBPETokenizer(
            add_prefix_space=add_prefix_space,
            lowercase=lowercase,
            dropout=dropout,
            unicode_normalizer=unicode_normalizer,
            continuing_subword_prefix=continuing_subword_prefix,
            end_of_word_suffix=end_of_word_suffix,
            trim_offsets=trim_offsets,
        )

    def train(
        self,
        files: List[Text],
        vocab_size: int,
        special_tokens: List[Token] = [],
        min_frequency: int = 2,
        **extras: Any,
    ) -> None:
        # pylint: disable=dangerous-default-value, arguments-differ
        # pylint: disable=too-many-arguments
        del extras  # use only required params

        # prepare special tokens with an initial set
        special_tokens = self.special_tokens + special_tokens

        # train a tokenizer model
        self.tokenizer.train(
            files,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens,
        )

    def save_pretrained(self, tokenizer_output_path: Union[Text, Path]) -> None:
        # make sure the `tokenizer_output_path` is a pathlike object
        if isinstance(tokenizer_output_path, str):
            tokenizer_output_path = Path(tokenizer_output_path)

        # save the trained tokenizer to `tokenizer_output_path`
        save_dir = path_to_posix(tokenizer_output_path)
        self.tokenizer.save_model(save_dir)

        # prepare the pre-trained tokenizer
        fast_tokenizer = TransformerTokenizerFast(
            vocab_file=path_to_posix(tokenizer_output_path / "vocab.json"),
            merges_file=path_to_posix(tokenizer_output_path / "merges.txt"),
            add_prefix_space=self.add_prefix_space,
            trim_offsets=self.trim_offsets,
            lowercase=self.lowercase,
            dropout=self.dropout,
        )

        # save the pre-trained tokenizer
        fix_tokenizer(fast_tokenizer)
        fast_tokenizer.save_pretrained(save_dir)

    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # fmt: off
        parser.add_argument("--files", type=str, nargs="+", default=None, required=True,
                            help="")
        parser.add_argument("--vocab_size", type=int, default=None, required=True,
                            help="")
        parser.add_argument("--tokenizer_output_path", type=str, default=None, required=True,
                            help="")
        # fmt: on
        return parser


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
        # pylint: disable=too-many-arguments
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

    def _batch_encode_plus(
        self, is_pretokenized: bool = False, **kwargs: Any
    ) -> BatchEncoding:
        # pylint: disable=arguments-differ
        assert self.add_prefix_space or not is_pretokenized, (
            f"You need to instantiate {self.__class__.__name__} with add_prefix_space=True "
            "to use it with pretokenized inputs."
        )

        return super()._batch_encode_plus(**kwargs)

    def _encode_plus(
        self, is_pretokenized: bool = False, **kwargs: Any
    ) -> BatchEncoding:
        # pylint: disable=arguments-differ
        assert self.add_prefix_space or not is_pretokenized, (
            f"You need to instantiate {self.__class__.__name__} with add_prefix_space=True "
            "to use it with pretokenized inputs."
        )

        return super()._encode_plus(**kwargs)

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

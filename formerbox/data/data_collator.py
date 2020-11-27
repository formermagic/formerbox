from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Text, Tuple, Union

import torch
from torch import Tensor
from torch.distributions.uniform import Uniform
from transformers import PreTrainedTokenizerFast
from typing_extensions import Literal

EncodedInput = Union[List[int], Tensor]
ReplaceLength = Literal[0, 1, -1]


def tolist(sequences: Union[List[Any], Tensor]) -> List[Any]:
    if isinstance(sequences, Tensor):
        return sequences.tolist()
    return sequences


def collate_batch(
    sequences: List[EncodedInput],
    tokenizer: PreTrainedTokenizerFast,
    pad_value: Optional[int] = None,
) -> Tensor:
    max_length = 0
    padding_required = False
    for idx, sequence in enumerate(sequences):
        if isinstance(sequence, list):
            sequence = torch.tensor(sequence)
            sequences[idx] = sequence

        # find max_length in sequences
        sequence_length = sequence.size(0)
        if sequence_length > max_length:
            max_length = sequence_length

        # check if sequences are of different lengths
        if sequence_length != max_length:
            padding_required = True

    # handle edge case when sequences have the same length
    if not padding_required:
        return torch.stack(sequences, dim=0)

    # prepare the result tensor with filled pad token id
    if pad_value is None:
        assert tokenizer.pad_token_id is not None
        pad_value = tokenizer.pad_token_id

    result = torch.LongTensor().new_full(
        size=[len(sequences), max_length],
        fill_value=pad_value,
    )

    # fill result tensor with sequences
    for idx, sequence in enumerate(sequences):
        assert isinstance(sequence, Tensor)
        padding_side = getattr(tokenizer, "padding_side", "right")
        sequence_length = sequence.size(0)
        if padding_side == "right":
            result[idx, :sequence_length] = sequence
        else:
            result[idx, -sequence_length:] = sequence

    return result


def find(tensor: Tensor, values: Tensor) -> Tensor:
    return torch.nonzero(tensor[:, None] == values)  # type: ignore


@dataclass
class DataCollatorForSeq2Seq:
    tokenizer: PreTrainedTokenizerFast

    def __call__(self, features: List[Dict[Text, EncodedInput]]) -> Dict[Text, Tensor]:
        assert self.tokenizer.pad_token_id is not None

        src_input_ids: List[EncodedInput] = []
        tgt_input_ids: List[EncodedInput] = []
        for feature in features:
            src_input_ids.append(feature["src_input_ids"])
            tgt_input_ids.append(feature["tgt_input_ids"])

        input_ids = collate_batch(sequences=src_input_ids, tokenizer=self.tokenizer)
        labels = collate_batch(sequences=tgt_input_ids, tokenizer=self.tokenizer)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


@dataclass
class DataCollatorForSeq2SeqDenoising:
    tokenizer: PreTrainedTokenizerFast
    masked_token_ratio: float = 0.15
    random_token_ratio: float = 0
    replace_length: ReplaceLength = -1

    def __call__(self, features: List[Dict[Text, EncodedInput]]) -> Dict[Text, Tensor]:
        assert self.tokenizer.pad_token_id is not None

        inputs: List[EncodedInput] = []
        for feature in features:
            input_ids = feature["src_input_ids"]
            inputs.append(tolist(input_ids))

        input_ids, labels = self.add_whole_word_mask(inputs)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def add_whole_word_mask(self, inputs: List[EncodedInput]) -> Tuple[Tensor, Tensor]:
        # pipeline: ids -> text -> encoding batch
        # we need these operations to get word bounds
        input_tokens = self.tokenizer.batch_decode(
            sequences=tolist(inputs),
            skip_special_tokens=True,
        )
        encoding_batch = self.tokenizer(input_tokens)

        # batched inputs for further collation
        batched_input_ids: List[EncodedInput] = []
        batched_labels: List[EncodedInput] = []

        for idx, input_ids in enumerate(encoding_batch.input_ids):
            # get word bounds for the given index
            input_words = encoding_batch.words(idx)

            # prepare masked source and labels
            source, labels = self.mask_words(input_ids, input_words)
            # append values to the accumulated lists
            batched_input_ids.append(source)
            batched_labels.append(labels)

        # collate accumulated tensors
        input_ids = collate_batch(sequences=batched_input_ids, tokenizer=self.tokenizer)
        labels = collate_batch(sequences=batched_labels, tokenizer=self.tokenizer)

        return input_ids, labels

    def mask_words(
        self, input_ids: List[int], input_words: List[Optional[int]]
    ) -> Tuple[Tensor, Tensor]:
        assert self.tokenizer.mask_token_id is not None

        # prepare the word boundaries for masking out
        word_bounds = torch.tensor([x if x is not None else -1 for x in input_words])
        word_ids = word_bounds[word_bounds != -1].unique()

        # prepare input and input mask to split masked subwords
        inputs = torch.tensor(input_ids)
        inputs_mask = torch.full(word_bounds.shape, fill_value=1, dtype=torch.bool)

        # prepare the target labels
        labels = inputs.clone()

        # pick random words to be masked out
        indices_masked = torch.full(
            size=word_ids.shape,
            fill_value=self.masked_token_ratio,
            dtype=torch.float,
        )

        indices_masked = torch.bernoulli(indices_masked).bool()
        masked_word_ids = word_ids[indices_masked]

        # pick random words to be swapped with a random token
        random_token_mask = (
            Uniform(0, 1).sample(masked_word_ids.shape) < self.random_token_ratio
        )

        # indices for tokens to mask out
        masked_tokens_indices = find(word_bounds, masked_word_ids)
        masked_tokens_indices = masked_tokens_indices[:, 0]
        # indices for tokens to replace with a random one
        random_tokens_indices = find(word_bounds, masked_word_ids[random_token_mask])
        random_tokens_indices = random_tokens_indices[:, 0]

        if masked_tokens_indices.nelement() == 0:
            return inputs, labels

        # cut off all masked out and random tokens
        if self.replace_length == 0:
            inputs_mask[masked_tokens_indices] = 0
        # replace each subword with a mask or random token
        # `Trans #form #er #s are great` => `<mask> <mask> <mask> <mask> are great`
        elif self.replace_length == -1:
            inputs[masked_tokens_indices] = self.tokenizer.mask_token_id
            inputs[random_tokens_indices] = self.random_tokens(
                shape=random_tokens_indices.shape
            )
        # replace each word with a mask or random token
        # `Trans #form #er #s are great` => `<mask> are great`
        else:
            # get a mask to find where words start
            masked_words = word_bounds[masked_tokens_indices].tolist()
            masked_words_start_mask = torch.tensor(self.words_start_mask(masked_words))

            # get indices for replacing word starts with mask tokens
            indices_replaced = masked_tokens_indices[masked_words_start_mask]
            # get indices for replacing word starts with a random token
            indices_random = indices_replaced[random_token_mask]

            # replace first non-repeating subwords with a mask or random token
            inputs[indices_replaced] = self.tokenizer.mask_token_id
            inputs[indices_random] = self.random_tokens(shape=indices_random.shape)

            # include only word starts for the final result
            inputs_mask[masked_tokens_indices] = masked_words_start_mask

        # get the final input ids
        inputs = inputs[inputs_mask]

        return inputs, labels

    def words_start_mask(self, elements: List[Any]) -> List[bool]:
        """Returns the mask indicating where words start."""
        result_mask: List[bool] = []
        prev_element: Optional[Any] = None

        for element in elements:
            if element != prev_element:
                result_mask.append(True)
            else:
                result_mask.append(False)

            prev_element = element

        return result_mask

    def random_tokens(self, shape: torch.Size) -> Tensor:
        """Returns random tokens sampled from the tokenizer vocab."""
        return torch.randint(self.tokenizer.vocab_size, shape).long()

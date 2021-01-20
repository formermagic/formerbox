import math
from abc import ABCMeta, abstractmethod
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


def tensors(sequences: List[Any]) -> List[Tensor]:
    result: List[Tensor] = []
    for sequence in sequences:
        if isinstance(sequence, Tensor):
            result.append(sequence)
        else:
            result.append(torch.tensor(sequence))
    return result


def collate_batch(
    sequences: List[EncodedInput],
    tokenizer: PreTrainedTokenizerFast,
    pad_value: Optional[int] = None,
) -> Tensor:
    # return an empty tensor if no examples provided
    if len(sequences) == 0:
        return torch.LongTensor()

    max_length = 0
    prev_max_length = 0
    padding_required = False
    for idx, sequence in enumerate(sequences):
        if isinstance(sequence, list):
            sequence = torch.tensor(sequence)
            sequences[idx] = sequence

        # find max_length in sequences
        sequence_length = sequence.size(0)
        if sequence_length > max_length:
            prev_max_length = max_length
            max_length = sequence_length

        # check if sequences are of different lengths
        if prev_max_length != max_length:
            padding_required = True

    # handle edge case when sequences have the same length
    if not padding_required:
        return torch.stack(tensors(sequences), dim=0)

    # prepare the result tensor with filled pad token id
    if pad_value is None:
        assert tokenizer.pad_token_id is not None
        pad_value = tokenizer.pad_token_id

    result = torch.full(
        size=[len(sequences), max_length],
        fill_value=pad_value,
        dtype=torch.long,
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
    return torch.nonzero(tensor[:, None] == values, as_tuple=False)  # type: ignore


@dataclass
class DataCollator(metaclass=ABCMeta):
    tokenizer: PreTrainedTokenizerFast

    @abstractmethod
    def __call__(self, features: List[Dict[Text, EncodedInput]]) -> Dict[Text, Tensor]:
        raise NotImplementedError()

    def _get_special_tokens_mask(
        self,
        input_ids: Union[List[int], Tensor],
        already_has_special_tokens: bool = True,
    ) -> torch.BoolTensor:
        """Returns the mask indicating special tokens in the given list."""
        special_tokens_mask = self.tokenizer.get_special_tokens_mask(
            token_ids_0=tolist(input_ids),
            already_has_special_tokens=already_has_special_tokens,
        )

        return torch.BoolTensor(special_tokens_mask)

    def _random_tokens(self, shape: torch.Size) -> Tensor:
        """Returns random tokens sampled from the tokenizer vocab."""
        return torch.randint(self.tokenizer.vocab_size, shape).long()


@dataclass
class DataCollatorForDenoising(DataCollator):
    masked_token_ratio: float = 0.15
    random_token_ratio: float = 0

    @abstractmethod
    def add_noise(self, inputs: List[EncodedInput]) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError()


@dataclass
class DataCollatorForTranslation(DataCollator):
    def __call__(self, features: List[Dict[Text, EncodedInput]]) -> Dict[Text, Tensor]:
        assert self.tokenizer.pad_token_id is not None

        seq_input_ids: List[EncodedInput] = []
        seq_labels: List[EncodedInput] = []
        for feature in features:
            seq_input_ids.append(feature["input_ids"])
            seq_labels.append(feature["labels"])

        input_ids = collate_batch(sequences=seq_input_ids, tokenizer=self.tokenizer)
        labels = collate_batch(sequences=seq_labels, tokenizer=self.tokenizer)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


@dataclass
class DataCollatorForWholeWordMasking(DataCollatorForDenoising):
    replace_length: ReplaceLength = -1

    def __call__(self, features: List[Dict[Text, EncodedInput]]) -> Dict[Text, Tensor]:
        assert self.tokenizer.pad_token_id is not None

        inputs: List[EncodedInput] = []
        for feature in features:
            input_ids = feature["input_ids"]
            inputs.append(tolist(input_ids))

        input_ids, labels = self.add_noise(inputs)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def add_noise(self, inputs: List[EncodedInput]) -> Tuple[Tensor, Tensor]:
        # pipeline: ids -> text -> encoding batch
        # we need these operations to get word bounds
        input_text = self.tokenizer.batch_decode(
            sequences=tolist(inputs),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,  # makes operation idempotent
        )

        encoding_batch = self.tokenizer(
            text=input_text,
            add_special_tokens=False,
            truncation=True,
        )

        # batched inputs for further collation
        batched_input_ids: List[EncodedInput] = []
        batched_labels: List[EncodedInput] = []

        for idx, input_ids in enumerate(encoding_batch.input_ids):
            # get word bounds for the given index
            input_words = encoding_batch.words(idx)

            # prepare masked source and labels
            source, labels = self.add_whole_word_mask(input_ids, input_words)
            # append values to the accumulated lists
            batched_input_ids.append(source)
            batched_labels.append(labels)

        # collate accumulated tensors
        input_ids = collate_batch(sequences=batched_input_ids, tokenizer=self.tokenizer)
        labels = collate_batch(sequences=batched_labels, tokenizer=self.tokenizer)

        return input_ids, labels

    def add_whole_word_mask(
        self, input_ids: List[int], input_words: List[Optional[int]]
    ) -> Tuple[Tensor, Tensor]:
        assert self.tokenizer.mask_token_id is not None

        # prepare the word boundaries for masking out
        special_tokens_mask = self._get_special_tokens_mask(input_ids)
        word_bounds = torch.tensor([x if x is not None else -1 for x in input_words])
        word_ids = word_bounds[~special_tokens_mask].unique()

        # prepare input and input mask to split masked subwords
        inputs = torch.tensor(input_ids)
        inputs_mask = torch.full(inputs.shape, fill_value=1, dtype=torch.bool)

        # prepare the target labels
        labels = inputs.clone()

        # pick random words to be masked out
        indices_masked = torch.full(word_ids.shape, float(self.masked_token_ratio))
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
            inputs[random_tokens_indices] = self._random_tokens(
                shape=random_tokens_indices.shape,  # type: ignore
            )
        # replace each word with a mask or random token
        # `Trans #form #er #s are great` => `<mask> are great`
        else:
            # get a mask to find where words start
            masked_words = word_bounds[masked_tokens_indices].tolist()
            masked_words_start_mask = torch.tensor(self._words_start_mask(masked_words))

            # get indices for replacing word starts with mask tokens
            indices_replaced = masked_tokens_indices[masked_words_start_mask]
            # get indices for replacing word starts with a random token
            indices_random = indices_replaced[random_token_mask]

            # replace first non-repeating subwords with a mask or random token
            inputs[indices_replaced] = self.tokenizer.mask_token_id
            inputs[indices_random] = self._random_tokens(
                shape=indices_random.shape,  # type: ignore
            )

            # include only word starts for the final result
            inputs_mask[masked_tokens_indices] = masked_words_start_mask

        # get the final input ids
        inputs = inputs[inputs_mask]

        return inputs, labels

    def _words_start_mask(self, elements: List[Any]) -> List[bool]:
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


@dataclass
class DataCollatorForBartDenoising(DataCollatorForDenoising):
    lambda_coef: float = 3.0

    def __call__(self, features: List[Dict[Text, EncodedInput]]) -> Dict[Text, Tensor]:
        assert self.tokenizer.pad_token_id is not None

        inputs: List[EncodedInput] = []
        for feature in features:
            input_ids = feature["input_ids"]
            inputs.append(input_ids)

        input_ids, labels = self.add_noise(inputs)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    def add_noise(self, inputs: List[EncodedInput]) -> Tuple[Tensor, Tensor]:
        # batched inputs for further collation
        batched_input_ids: List[EncodedInput] = []
        batched_labels: List[EncodedInput] = []

        for input_ids in inputs:
            # convert lists to tensors
            if isinstance(input_ids, list):
                input_ids = torch.tensor(input_ids)

            # prepare masked input ids and labels
            special_tokens_mask = self._get_special_tokens_mask(input_ids)
            input_ids, labels = self.add_infiling_noise(input_ids, special_tokens_mask)

            # append values to the accumulated lists
            batched_input_ids.append(input_ids)
            batched_labels.append(labels)

        # collate accumulated tensors
        input_ids = collate_batch(sequences=batched_input_ids, tokenizer=self.tokenizer)
        labels = collate_batch(sequences=batched_labels, tokenizer=self.tokenizer)

        return input_ids, labels

    def add_insertion_noise(
        self, tokens: Tensor, special_tokens_mask: Tensor, insertion_prob: float
    ) -> Tuple[Tensor, Tensor]:
        assert self.tokenizer.mask_token_id is not None

        # count the number of insertions to make
        tokens_mask = ~special_tokens_mask
        num_tokens = tokens_mask.long().sum()
        num_insertions = math.ceil(num_tokens * insertion_prob)

        # select indices use for inserting mask tokens
        insertions = torch.LongTensor(num_insertions).fill_(1)
        insertions_prob = torch.cat((tokens_mask.long(), insertions), dim=0).float()
        insertions_indices = torch.multinomial(
            insertions_prob, num_samples=num_insertions, replacement=False
        )

        # a mask for tokens to be inserted
        insertion_mask = torch.LongTensor(tokens.size(0) + num_insertions).fill_(0)
        insertion_mask[insertions_indices] = 1
        insertion_mask = insertion_mask.bool()

        # a mask for tokens to be random among inserted indices
        random_prob = torch.full(insertion_mask.shape, float(self.random_token_ratio))
        random_mask = torch.bernoulli(random_prob).bool()
        random_size = tokens[insertion_mask & random_mask].size()

        # prepare the input tensor with insertions of mask and random tokens
        input_ids = torch.LongTensor(tokens.size(0) + num_insertions).fill_(0)
        input_ids[insertion_mask] = self.tokenizer.mask_token_id
        input_ids[insertion_mask & random_mask] = self._random_tokens(random_size)  # type: ignore
        input_ids[~insertion_mask] = tokens

        # prepare the target tensor
        labels = tokens.clone()

        return input_ids, labels

    def add_infiling_noise(
        self, inputs: Tensor, special_tokens_mask: Tensor
    ) -> Tuple[Tensor, Tensor]:
        assert self.tokenizer.mask_token_id is not None

        # count the number of maskable tokens
        tokens_mask = ~special_tokens_mask
        num_tokens = tokens_mask.long().sum().item()

        # handle 0-tokens masking
        if num_tokens == 0:
            return inputs.clone(), inputs.clone()

        # sample mask spans with the given masked token ratio
        num_to_mask = math.ceil(float(num_tokens) * self.masked_token_ratio)
        mask_spans = self._sample_mask_spans(num_to_mask)
        # update the actual number of tokens to mask
        num_inserts = num_to_mask - mask_spans.size(0)
        num_to_mask = num_to_mask - num_inserts

        # add insertion noise if 0-length span sampled
        if num_to_mask == 0:
            insertion_prob = num_inserts / num_tokens
            return self.add_insertion_noise(
                inputs,
                insertion_prob=insertion_prob,
                special_tokens_mask=special_tokens_mask,
            )

        # sample masking spans start indices
        mask_indices = torch.multinomial(
            tokens_mask.float(),
            num_samples=num_to_mask,
            replacement=False,
        )

        # mask to replace tokens with a mask token
        replace_mask = torch.zeros_like(inputs)
        # mask to keep original tokens in the output
        original_mask = torch.ones_like(inputs)

        for span_start, span_length in zip(mask_indices, mask_spans):
            # num_tokens + 1 ensures that we mask at least one token
            span_end = min(span_start + span_length, inputs.size(0))
            span_length = span_end - span_start

            # get span special tokens indices
            span_tokens = special_tokens_mask[span_start:span_end]
            span_indices = torch.nonzero(span_tokens, as_tuple=False)  # type: ignore
            span_indices = span_indices[:, 0]

            # keep tokens succeeding special tokens to split spaces into parts
            # clamping prevents getting out of the span bounds
            span_indices = (span_indices + 1).clamp_max(span_length - 1)

            # mask for span tokens to keep in the result
            span_keep_mask = torch.zeros_like(span_tokens)
            # keep span special tokens in noised result
            span_keep_mask[span_indices] = 1
            # keep span start token as it gets masked
            span_keep_mask[0] = 1

            # prepare span mask for tokens to get replaced
            replace_mask[span_start:span_end] = 1
            # prepare span mask for tokens to keep in the result
            original_mask[span_start:span_end] = span_keep_mask

        # avoid replacing special tokens
        replace_mask = replace_mask.bool() & tokens_mask
        original_mask.masked_fill_(special_tokens_mask, value=1)
        original_mask = original_mask.bool()

        # prepare the input tensor with masked spans
        input_ids = inputs.clone()
        input_ids[replace_mask] = self.tokenizer.mask_token_id
        input_ids = input_ids[original_mask]

        # prepare the target tensor
        labels = inputs.clone()

        return input_ids, labels

    def _sample_mask_spans(self, num_to_mask: int) -> Tensor:
        """Returns the mask spans sampled from poisson PMF with given number of tokens to mask."""

        # handle empty masking case
        if num_to_mask == 0:
            return torch.tensor([])

        # sample masking spans from poisson prob mass function
        lengths = self._sample_poisson(num_to_mask)

        # make sure spans are long enough to take effect from masking
        cum_lengths = lengths.cumsum(0)
        while cum_lengths[-1] < num_to_mask:
            span_lengths = self._sample_poisson(num_to_mask)
            lengths = torch.cat((lengths, span_lengths), dim=0)
            cum_lengths = lengths.cumsum(0)

        # find upper bound span index for further trimming
        last_span_index = 0
        while cum_lengths[last_span_index] < num_to_mask:
            last_span_index += 1

        # trim upper bound span mask
        if last_span_index != 0:
            num_masked = cum_lengths[last_span_index - 1]
            lengths[last_span_index] = num_to_mask - num_masked
        else:
            lengths[last_span_index] = num_to_mask

        # update the number of spans to mask
        num_to_mask = last_span_index + 1
        # get spans subset of desired length
        lengths = lengths[:num_to_mask]
        # take only non-zero spans
        lengths = lengths[lengths > 0]

        return lengths

    def _sample_poisson(self, length: int) -> Tensor:
        """Returns a sample from poisson probability mass function of given length."""
        return torch.poisson(torch.FloatTensor(length).fill_(self.lambda_coef)).long()

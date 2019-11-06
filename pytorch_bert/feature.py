import random
from collections import namedtuple
from typing import List, Tuple, Union, cast

import torch

from .tokenizer import SpecialToken, SubWordTokenizer

SequencePair = Tuple[str, str]
Sequences = Union[Tuple[str], SequencePair]
Feature = namedtuple("Feature", ("tokens", "input_type_ids", "input_ids", "input_mask"))
Masked = namedtuple("Masked", ("positions", "answers"))


def convert_sequences_to_feature(
    tokenizer: SubWordTokenizer, sequences: Sequences, max_sequence_length: int
) -> Feature:
    tokenized_sequences = tuple(tokenizer.tokenize(sequence) for sequence in sequences)
    is_sequence_pair = _is_sequence_pair(tokenized_sequences)

    if is_sequence_pair:
        # [CLS], sequence1, [SEP], sequence2, [SEP]
        tokenized_sequences = cast(Tuple[List[str], List[str]], tokenized_sequences)
        tokenized_sequences = _truncate_sequence_pair(tokenized_sequences, max_sequence_length - 3)
    else:
        # [CLS], sequence1, [SEP]
        if len(tokenized_sequences[0]) > max_sequence_length - 2:
            tokenized_sequences = tuple(tokenized_sequences[0][0 : max_sequence_length - 2])

    tokens = [SpecialToken.cls_] + tokenized_sequences[0] + [SpecialToken.sep]
    input_type_ids = [0] * (len(tokenized_sequences[0]) + 2)

    if is_sequence_pair:
        tokens.extend(tokenized_sequences[1])
        tokens.append(SpecialToken.sep)

        input_type_ids.extend([1] * (len(tokenized_sequences[1]) + 1))

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    if len(input_ids) < max_sequence_length:
        list_for_padding = [0] * (max_sequence_length - len(input_ids))

        input_type_ids.extend(list_for_padding)
        input_ids.extend(list_for_padding)
        input_mask.extend(list_for_padding)

    return Feature(tokens, input_type_ids, input_ids, input_mask)


def create_input_mask(input_mask: List[List[int]], max_sequence_length: int):
    return torch.ones((len(input_mask), max_sequence_length), dtype=torch.bool) ^ torch.tensor(
        input_mask, dtype=torch.bool
    )


def mask(feature: Feature, mask_token_id: int):
    tokens, input_type_ids, input_ids, input_mask = feature
    masked_positions = []
    answers = []

    for index, token in enumerate(tokens):
        if token == SpecialToken.cls_ or token == SpecialToken.sep:
            continue

        if random.random() < 0.15:
            masked_positions.append(index)
            answers.append(input_ids[index])

            tokens[index] = SpecialToken.mask
            input_ids[index] = mask_token_id

    return Feature(tokens, input_type_ids, input_ids, input_mask), Masked(masked_positions, answers)


def _is_sequence_pair(sequences: Tuple) -> bool:
    return len(sequences) == 2


def _truncate_sequence_pair(
    tokenized_sequences: Tuple[List[str], List[str]], max_length: int
) -> Tuple[List[str], List[str]]:
    sequence1, sequence2 = tokenized_sequences

    while True:
        total_length = len(sequence1) + len(sequence2)
        if total_length <= max_length:
            return (sequence1, sequence2)

        if len(sequence1) > len(sequence2):
            sequence1.pop()
        else:
            sequence2.pop()

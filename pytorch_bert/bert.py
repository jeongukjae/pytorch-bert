import itertools
from typing import cast, Union, Tuple, List

from .tokenizer import SubWordTokenizer, SpecialToken

SequencePair = Tuple[str, str]
Sequences = Union[Tuple[str], SequencePair]


class FeatureExtractor:
    def __init__(self, tokenizer: SubWordTokenizer):
        self.tokenizer = tokenizer

    def convert_sequences_to_feature(
        self, sequences: Sequences, max_sequence_length: int
    ) -> Tuple[List[str], List[int], List[int], List[int]]:
        tokenized_sequences = tuple(self.tokenizer.tokenize(sequence) for sequence in sequences)
        is_sequence_pair = _is_sequence_pair(tokenized_sequences)

        if is_sequence_pair:
            # [CLS], sequence1, [SEP], sequence2, [SEP]
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

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        if len(input_ids) < max_sequence_length:
            list_for_padding = [0] * (max_sequence_length - len(input_ids))

            input_type_ids.extend(list_for_padding)
            input_ids.extend(list_for_padding)
            input_mask.extend(list_for_padding)

        return (tokens, input_type_ids, input_ids, input_mask)


def _is_sequence_pair(sequences: Sequences) -> bool:
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

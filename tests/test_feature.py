import math

import pytest

import pytorch_bert.feature as F
import pytorch_bert.tokenizer as T

VOCAB_WORDS = [
    "[PAD]",
    "[UNK]",
    "[SEP]",
    "[CLS]",
    "[MASK]",
    "the",
    "dog",
    "is",
    "hairy",
    ",",
    "this",
    "jack",
    "##son",
    "##ville",
    "?",
    "no",
    "it",
    "not",
]


@pytest.mark.parametrize(
    "tokenized_sequences,max_length,expected_output",
    [
        pytest.param(
            (
                ["the", "dog", "is", "hairy", ",", "is", "this", "jack", "##son", "##ville", "?"],
                ["no", "it", "is", "not"],
            ),
            12,
            (["the", "dog", "is", "hairy", ",", "is", "this", "jack"], ["no", "it", "is", "not"]),
        ),
        pytest.param(
            (["the", "dog", "is", "hairy"], ["is", "this", "jack", "##son", "##ville", "?", "no", "it", "is", "not"]),
            12,
            (["the", "dog", "is", "hairy"], ["is", "this", "jack", "##son", "##ville", "?", "no", "it"]),
        ),
    ],
)
def test_truncate_sequence_pair(tokenized_sequences, max_length, expected_output):
    assert F._truncate_sequence_pair(tokenized_sequences, max_length) == expected_output


@pytest.mark.parametrize(
    "sequence,max_sequence_length,expected_output",
    [
        # fmt: off
        pytest.param(
            ("the dog is hairy, is this jacksonville?", "no it is not"),
            18,
            (
                ["[CLS]", "the", "dog", "is", "hairy", ",", "is", "this", "jack", "##son", "##ville", "?", "[SEP]", "no", "it", "is", "not", "[SEP]"],
                [3,       5,     6,     7,    8,       9,   7,    10,     11,     12,      13,        14,  2,       15,   16,   7,    17,    2],
                [0,       0,     0,     0,    0,       0,   0,     0,      0,      0,       0,         0,  0,       1,    1,    1,    1,     1],
                [0,       0,     0,     0,    0,       0,   0,     0,      0,      0,       0,         0,  0,       0,    0,    0,    0,     0]
            )
        ),
        pytest.param(
            ("the dog is hairy, is this jacksonville?", "no it is not"),
            20,
            (
                ["[CLS]", "the", "dog", "is", "hairy", ",", "is", "this", "jack", "##son", "##ville", "?", "[SEP]", "no", "it", "is", "not", "[SEP]"],
                [3,       5,     6,     7,    8,       9,   7,    10,     11,     12,      13,        14,  2,       15,   16,   7,    17,    2,      0,  0],
                [0,       0,     0,     0,    0,       0,   0,     0,      0,      0,       0,         0,  0,       1,    1,    1,    1,     1,      0,  0],
                [0,       0,     0,     0,    0,       0,   0,     0,      0,      0,       0,         0,  0,       0,    0,    0,    0,     0,      -math.inf, -math.inf]
            ),
        ),
        pytest.param(
            ("the dog is hairy, is this jacksonville?",),
            20,
            (
                ["[CLS]", "the", "dog", "is", "hairy", ",", "is", "this", "jack", "##son", "##ville", "?", "[SEP]"],
                [3,       5,     6,     7,    8,       9,   7,    10,     11,     12,      13,        14,  2,        0,  0,  0,  0,  0,  0,  0],
                [0,       0,     0,     0,    0,       0,   0,     0,      0,      0,       0,         0,  0,        0,  0,  0,  0,  0,  0,  0],
                [0,       0,     0,     0,    0,       0,   0,     0,      0,      0,       0,         0,  0,       -math.inf, -math.inf, -math.inf, -math.inf, -math.inf, -math.inf, -math.inf]
            )
        )
        # fmt: on
    ],
)
def test_convert_sequences_to_feature(tmpdir, sequence, max_sequence_length, expected_output):
    vocab_path = tmpdir.join("test-vocab-file.txt")
    vocab_path.write("\n".join(VOCAB_WORDS))

    vocab = T.Vocab(vocab_path)
    tokenizer = T.SubWordTokenizer(vocab, True)
    output = F.convert_sequences_to_feature(tokenizer, sequence, max_sequence_length)

    assert output == expected_output

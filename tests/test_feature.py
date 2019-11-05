import pytest

import pytorch_bert.feature as F


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

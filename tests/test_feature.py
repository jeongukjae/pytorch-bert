import pytest

from pytorch_bert.feature import _truncate_sequence_pair


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
    assert _truncate_sequence_pair(tokenized_sequences, max_length) == expected_output

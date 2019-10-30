from pytorch_bert.tokenizer import Vocab


def test_load_vocab(tmpdir):
    path = tmpdir.mkdir("test").join("vocab.txt")
    path.write("\n".join(["word1", "word2", "word3"]))

    vocab = Vocab(str(path))

    assert vocab.convert_id_to_token(0) == "word1"
    assert vocab.convert_token_to_id("word2") == 1
    assert vocab.convert_ids_to_tokens([1, 0]) == ["word2", "word1"]
    assert vocab.convert_tokens_to_ids(["word3", "word1"]) == [2, 0]

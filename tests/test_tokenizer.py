import pytorch_bert.tokenizer as T


def test_load_vocab(tmpdir):
    path = tmpdir.mkdir("test").join("vocab.txt")
    path.write("\n".join(["word1", "word2", "word3"]))

    vocab = T.Vocab(str(path))

    assert vocab.convert_id_to_token(0) == "word1"
    assert vocab.convert_token_to_id("word2") == 1
    assert vocab.convert_ids_to_tokens([1, 0]) == ["word2", "word1"]
    assert vocab.convert_tokens_to_ids(["word3", "word1"]) == [2, 0]


def test_full_tokenizer(tmpdir):
    path = tmpdir.mkdir("test").join("full_vocab.txt")
    path.write("\n".join(["[UNK]", "[CLS]", "[SEP]", "want", "##want", "##ed", "wa", "un", "runn", "##ing", ","]))

    vocab = T.Vocab(path)
    tokenizer = T.SubWordTokenizer(vocab)

    tokens = tokenizer.tokenize("UNwant\u00E9d,running")

    assert tokens == ["un", "##want", "##ed", ",", "runn", "##ing"]
    assert tokenizer.convert_tokens_to_ids(tokens) == [7, 4, 5, 10, 8, 9]
    assert tokenizer.convert_ids_to_tokens([7, 4, 5, 10, 8, 9]) == tokens


def test_basic_tokenizer_no_lower():
    tokenizer = T.BasicTokenizer(do_lower_case=False)

    assert tokenizer.tokenize(" \tHeLLo!how  \n Are yoU?  ") == ["HeLLo", "!", "how", "Are", "yoU", "?"]


def test_basic_tokenizer_do_lower():
    lowered_tokenizer = T.BasicTokenizer(do_lower_case=True)

    assert lowered_tokenizer.tokenize(" \tHeLLo!how  \n Are yoU?  ") == ["hello", "!", "how", "are", "you", "?"]
    assert lowered_tokenizer.tokenize("H\u00E9llo") == ["hello"]


def test_basic_tokenizer_with_chinese_character():
    tokenizer = T.BasicTokenizer()

    assert tokenizer.tokenize("ah\u535A\u63A8zz") == ["ah", "\u535A", "\u63A8", "zz"]


def test_wordpiece_tokenizer(tmpdir):
    path = tmpdir.mkdir("test").join("vocab.txt")
    path.write("\n".join(["[UNK]", "[CLS]", "[SEP]", "want", "##want", "##ed", "wa", "un", "runn", "##ing"]))

    vocab = T.Vocab(str(path))
    tokenizer = T.WordpieceTokenizer(vocab=vocab)

    assert tokenizer.tokenize("") == []
    assert tokenizer.tokenize("unwanted running") == ["un", "##want", "##ed", "runn", "##ing"]
    assert tokenizer.tokenize("unwantedX running") == ["[UNK]", "runn", "##ing"]


def test_clean_text():
    assert T.clean_text("\tHello\n안녕안녕   mm") == " Hello 안녕안녕   mm"


def test_is_whitespace():
    assert T._is_whitespace(" ")
    assert T._is_whitespace("\t")
    assert T._is_whitespace("\r")
    assert T._is_whitespace("\n")
    assert T._is_whitespace("\u00A0")

    assert not T._is_whitespace("A")
    assert not T._is_whitespace("-")


def test_is_control():
    assert T._is_control("\u0005")

    assert not T._is_control("A")
    assert not T._is_control(" ")
    assert not T._is_control("\t")
    assert not T._is_control("\r")
    assert not T._is_control("\U0001F4A9")


def test_is_punctuation():
    assert T._is_punctuation("-")
    assert T._is_punctuation("$")
    assert T._is_punctuation("`")
    assert T._is_punctuation(".")

    assert not T._is_punctuation("A")
    assert not T._is_punctuation(" ")


def test_tokenize_chinese_chars():
    assert T._tokenize_chinese_chars("This is a Chinese character 一") == "This is a Chinese character  一 "
    assert T._tokenize_chinese_chars("no Chinese characters.") == "no Chinese characters."
    assert T._tokenize_chinese_chars("Some喥Rando喩mChi噟neseCharacter") == "Some 喥 Rando 喩 mChi 噟 neseCharacter"


def test_is_chinese_char():
    assert T._is_chinese_char(ord("一"))
    assert T._is_chinese_char(ord("壚"))
    assert not T._is_chinese_char(ord("a"))

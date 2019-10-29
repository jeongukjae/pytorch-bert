import unicodedata
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union, cast


class SpecialToken:
    unk = "[UNK]"
    sep = "[SEP]"
    cls_ = "[CLS]"
    mask = "[MASK]"


class SubWordTokenizer:
    def __init__(self, vocab: "Vocab", do_lower_case: bool = True):
        self.vocab = vocab

        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)

    def tokenize(self, text: str) -> List[str]:
        return [
            sub_token
            for token in self.basic_tokenizer.tokenize(text)
            for sub_token in self.wordpiece_tokenizer.tokenize(token)
        ]

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return self.vocab.convert_tokens_to_ids(tokens)

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        return self.vocab.convert_ids_to_tokens(ids)


class Vocab:
    def __init__(self, vocab_path: str):
        self.vocab = self._load_vocab(vocab_path)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def __contains__(self, key: str) -> bool:
        return key in self.vocab

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return cast(List[int], self._convert_by_vocab(self.vocab, tokens))

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        return cast(List[str], self._convert_by_vocab(self.inv_vocab, ids))

    @staticmethod
    def _load_vocab(vocab_path: str) -> OrderedDict:
        vocab = OrderedDict()
        index = 0
        with open(vocab_path, "r") as f:
            for line in f:
                token = _convert_to_str(line).strip()
                vocab[token] = index
                index += 1

        return vocab

    @staticmethod
    def _convert_by_vocab(vocab: Dict, items: List[Union[int, str]]) -> List[Union[int, str]]:
        return [vocab[item] for item in items]


class BasicTokenizer(object):
    """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

    def __init__(self, do_lower_case=True):
        """Constructs a BasicTokenizer.
        Args:
            do_lower_case: Whether to lower case the input.
        """
        self.do_lower_case = do_lower_case

    def tokenize(self, text: str) -> List[str]:
        """Tokenizes a piece of text."""
        text = _clean_text(text)

        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        text = _tokenize_chinese_chars(text)

        original_tokens = _tokenize_whitespace(text)
        splitted_tokens = []
        for token in original_tokens:
            if self.do_lower_case:
                token = token.lower()
                token = self._strip_accents(token)
            splitted_tokens.extend(self._split_on_punc(token))

        output_tokens = _tokenize_whitespace(" ".join(splitted_tokens))
        return output_tokens

    def _strip_accents(self, text: str) -> str:
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = [char for char in text if unicodedata.category(char) != "Mn"]

        return "".join(output)

    def _split_on_punc(self, text: str) -> List[str]:
        """Splits punctuation on a piece of text."""
        start_new_word = True
        output = []

        for char in text:
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)

        return ["".join(x) for x in output]


class WordpieceTokenizer:
    """Runs WordPiece tokenziation."""

    __PREFIX_OF_SUBWORD = "##"

    def __init__(self, vocab: Vocab, unknown_token: str = SpecialToken.unk, max_length_of_word: int = 200):
        self.vocab = vocab
        self.unknown_token = unknown_token
        self.max_length_of_word = max_length_of_word

    def tokenize(self, text: str) -> List[str]:
        """Tokenizes a piece of text into its word pieces.
        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.
        For example:
            input = "unaffable"
            output = ["un", "##aff", "##able"]
        Args:
            text: A single token or whitespace separated tokens. This should have
                already been passed through `BasicTokenizer.
        Returns:
            A list of wordpiece tokens.
        """
        return [subword for token in _tokenize_whitespace(text) for subword in self._split_to_subwords(token)]

    def _split_to_subwords(self, token: str) -> List[str]:
        if len(token) > self.max_length_of_word:
            return [self.unknown_token]

        start = 0
        subwords = []

        while start < len(token):
            subword, end = self._find_subword_in_token(token, start)
            if subword is None:
                return [self.unknown_token]
            subwords.append(subword)
            start = end

        return subwords

    def _find_subword_in_token(self, token: str, start_position: int) -> Tuple[Optional[str], int]:
        end_position = len(token)
        while end_position > start_position:
            subword = token[start_position:end_position]
            if start_position > 0:
                subword = self.__PREFIX_OF_SUBWORD + subword

            if subword in self.vocab:
                return subword, end_position

            end_position -= 1

        return None, end_position


def _convert_to_str(text: Union[str, bytes]) -> str:
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))


def _tokenize_whitespace(text: str) -> List[str]:
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    return text.strip().split()


def _is_punctuation(char: str) -> bool:
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


def _clean_text(text: str) -> str:
    """Performs invalid character removal and whitespace cleanup on text."""
    output = [" " if _is_whitespace(char) else char for char in text if not _is_invalid_char(char)]
    return "".join(output)


def _is_whitespace(char: str) -> bool:
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    category_of_char = unicodedata.category(char)
    if category_of_char == "Zs":
        return True
    return False


def _is_invalid_char(char: str) -> bool:
    char_code = ord(char)

    return char_code == 0 or char_code == 0xFFFD or _is_control(char)


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat in ("Cc", "Cf"):
        return True
    return False


def _tokenize_chinese_chars(text):
    """Adds whitespace around any CJK character."""
    output = []
    for char in text:
        char_code = ord(char)
        if _is_chinese_char(char_code):
            output.extend([" ", char, " "])
        else:
            output.append(char)
    return "".join(output)


def _is_chinese_char(char_code: int):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if (
        (char_code >= 0x4E00 and char_code <= 0x9FFF)
        or (char_code >= 0x3400 and char_code <= 0x4DBF)  #
        or (char_code >= 0x20000 and char_code <= 0x2A6DF)  #
        or (char_code >= 0x2A700 and char_code <= 0x2B73F)  #
        or (char_code >= 0x2B740 and char_code <= 0x2B81F)  #
        or (char_code >= 0x2B820 and char_code <= 0x2CEAF)  #
        or (char_code >= 0xF900 and char_code <= 0xFAFF)
        or (char_code >= 0x2F800 and char_code <= 0x2FA1F)  #
    ):  #
        return True

    return False

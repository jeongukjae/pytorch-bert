from .tokenizer import SubWordTokenizer


def convert_inputs_to_feature(tokenizer: SubWordTokenizer, text: str):
    subwords = tokenizer.tokenize(text)
    pass

import json
from collections import namedtuple

from pytorch_bert.tokenizer import clean_text, tokenize_whitespace

SquadExample = namedtuple(
    "SquadExample", ("context_text", "question_text", "answer_text", "start_position", "end_position")
)


def read_squad_example(json_path: str):
    with open(json_path, "r") as f:
        paragraphs = [paragraph for data in json.load(f)["data"] for paragraph in data["paragraphs"]]

    examples = []

    for paragraph in paragraphs:
        original_context_text = paragraph["context"]
        context_text = clean_text(original_context_text)

        for qa in paragraph["qas"]:
            question_text = clean_text(qa["question"])
            answer = qa["answers"][0]
            answer_text = clean_text(answer["text"])

            start_position = context_text[:context_text.index(answer_text)].count(" ")
            end_position = start_position + answer_text.count(" ")

            examples.append(SquadExample(context_text, question_text, answer_text, start_position, end_position))

    return examples


def is_whitespace(character: str) -> bool:
    return character == " " or character == "\t" or character == "\r" or character == "\n" or ord(character) == 0x202F

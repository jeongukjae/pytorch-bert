import json
import os
from pytorch_bert.feature import create_input_mask
import urllib.request
import zipfile
from typing import List
from collections import namedtuple

import torch
from torch.utils.data import RandomSampler, TensorDataset

from pytorch_bert.tokenizer import clean_text
from pytorch_bert import convert_sequences_to_feature, BertConfig, Bert, Vocab, SubWordTokenizer

SquadExample = namedtuple(
    "SquadExample", ("context_text", "question_text", "answer_text", "start_position", "end_position")
)


def download_model_file(url: str, cache_directory: str = "/tmp", force_download: bool = False):
    filename = url.split("/")[-1]

    if not os.path.isdir(cache_directory):
        raise ValueError(f"{cache_directory} is not a directory")

    download_path = os.path.join(cache_directory, filename)
    if not force_download and os.path.exists(download_path):
        return

    urllib.request.urlretrieve(url, download_path)

    model_zip = zipfile.ZipFile(download_path)
    model_zip.extractall(cache_directory)
    model_zip.close()


def read_squad_example(json_path: str) -> List[SquadExample]:
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

            start_position = context_text[: context_text.index(answer_text)].count(" ")
            end_position = start_position + answer_text.count(" ")

            examples.append(SquadExample(context_text, question_text, answer_text, start_position, end_position))

    return examples


def prepare_dataset(examples: List[SquadExample], tokenizer: SubWordTokenizer, max_sequence_length: int):
    features = [
        convert_sequences_to_feature(tokenizer, (example.context_text, example.question_text), max_sequence_length)
        for example in examples
    ]

    input_type_ids_of_features = torch.tensor([feature.input_type_ids for feature in features])
    input_ids_of_features = torch.tensor([feature.input_ids for feature in features])
    input_mask_of_features = create_input_mask([feature.input_mask for feature in features], max_sequence_length)

    start_position_of_features = torch.tensor([example.start_position for example in examples])
    end_position_of_features = torch.tensor([example.end_position for example in examples])

    return TensorDataset(
        input_type_ids_of_features,
        input_ids_of_features,
        input_mask_of_features,
        start_position_of_features,
        end_position_of_features,
    )

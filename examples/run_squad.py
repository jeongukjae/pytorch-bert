import os
import urllib.request
import zipfile
import json

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from pytorch_bert import convert_sequences_to_feature, BertConfig, Bert, Vocab, SubWordTokenizer
from pytorch_bert.feature import create_input_mask, mask
from pytorch_bert.weight_converter import load_tf_weight_to_pytorch_bert


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


def is_whitespace(c: str) -> bool:
    return c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F


def read_squad(json_path: str):
    with open(json_path, "r") as f:
        paragraph = [paragraph for data in json.load(f)["data"] for paragraph in data["paragraphs"]]


class SquadDataLoader(DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        collate_fn=None,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
        multiprocessing_context=None,
    ):
        pass


class BertForSquad(nn.Module):
    def __init__(self, config: BertConfig):
        super(BertForSquad, self).__init__()
        self.bert = Bert(config)
        self.squad_layer = nn.Linear(config.hidden_size, 2)

        nn.init.normal_(self.squad_layer.weight, std=0.02)
        nn.init.zeros_(self.squad_layer.bias)

    def forward(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor, attention_mask: torch.Tensor):
        encoder_outputs, _ = self.bert(input_ids, token_type_ids, attention_mask)
        encoder_outputs = encoder_outputs.view(-1, config.hidden_size)

        logits = self.squad_layer(encoder_outputs)
        return logits


def get_index(logit: torch.Tensor):
    return logit.argsort(dim=0)


if __name__ == "__main__":
    download_model_file(
        "https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip",
        "/tmp/bert-base",
        force_download=False,
    )

    config = BertConfig.from_json("/tmp/bert-base/multi_cased_L-12_H-768_A-12/bert_config.json")
    model = BertForSquad(config)
    load_tf_weight_to_pytorch_bert(model.bert, config, "/tmp/bert-base/multi_cased_L-12_H-768_A-12/bert_model.ckpt")
    model.bert.eval()

    # vocab = Vocab("/tmp/bert-base/multi_cased_L-12_H-768_A-12/vocab.txt")
    # tokenizer = SubWordTokenizer(vocab)

    # feature = convert_sequences_to_feature(
    #     tokenizer,
    #     (
    #         "To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?",
    #         "Architecturally, the school has a Catholic character. Atop the Main Building’s gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend “Venite Ad Me Omnes”. Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.",
    #     ),
    #     config.max_position_embeddings,
    # )
    # print(feature.tokens)
    # input_ids = torch.tensor([feature.input_ids])
    # input_type_ids = torch.tensor([feature.input_type_ids])
    # input_mask = create_input_mask(feature.input_mask, config.max_position_embeddings)

    # logits = model(input_ids, input_type_ids, input_mask)
    # start_indices = get_index(logits[:, 0])
    # end_indices = get_index(logits[:, 1])

    # print(start_indices)
    # print(end_indices)

    # print(feature.tokens[start_indices[0].item() : end_indices[0].item()])

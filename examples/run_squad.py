import os
from multiprocessing import Pool

import torch
from torch import nn

from pytorch_bert import Bert, BertConfig, SubWordTokenizer
from pytorch_bert.weight_converter import load_tf_weight_to_pytorch_bert
from util_for_squad import download_model_file, prepare_dataset, read_squad_example


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
    num_processes = int(os.getenv("PYTORCH_BERT_NUM_PROCESSES", 8))
    pool = Pool(num_processes)

    print("start to download model file")
    download_model_file(
        "https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip",
        "/tmp/bert-base",
        force_download=False,
    )

    print("build BERT model")
    config = BertConfig.from_json("/tmp/bert-base/multi_cased_L-12_H-768_A-12/bert_config.json")
    model = BertForSquad(config)

    print("load BERT weight")
    load_tf_weight_to_pytorch_bert(model.bert, config, "/tmp/bert-base/multi_cased_L-12_H-768_A-12/bert_model.ckpt")

    print("prepare dataset")
    tokenizer = SubWordTokenizer("/tmp/bert-base/multi_cased_L-12_H-768_A-12/vocab.txt")

    print("read squad file")
    examples = read_squad_example("./data/train-v1.1.json", pool)

    print("convert squad example to dataset")
    dataset = prepare_dataset(examples, tokenizer, config.max_position_embeddings, pool)

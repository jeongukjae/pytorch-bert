# WIP
import json
import argparse
import logging
import sys
from typing import NamedTuple, List

import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, RandomSampler, DataLoader

from pytorch_bert import Bert, BertConfig, SubWordTokenizer
from pytorch_bert.tokenizer import clean_text
from pytorch_bert.weight_converter import load_tf_weight_to_pytorch_bert
from pytorch_bert.feature import convert_sequences_to_feature

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", required=True)
parser.add_argument("--config-path", required=True)
parser.add_argument("--vocab-path", required=True)
parser.add_argument("--squad-train-path", required=True)
parser.add_argument("--epoch", default=3, type=int)
parser.add_argument("--batch-size", default=32, type=int)
parser.add_argument("--learing-rate", default=2e-5, type=float)
parser.add_argument("--logging-step", default=200, type=int)
parser.add_argument("--eval-step", default=500, type=int)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
logger.addHandler(handler)


class BertForSquad(nn.Module):
    def __init__(self, config: BertConfig):
        super(BertForSquad, self).__init__()
        self.bert = Bert(config)
        self.squad_layer = nn.Linear(config.hidden_size, 2)

    def forward(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor, attention_mask: torch.Tensor):
        encoder_outputs, _ = self.bert(input_ids, token_type_ids, attention_mask)
        logits = self.squad_layer(encoder_outputs)
        return logits


class SquadExample(NamedTuple):
    context_text: str
    question_text: str
    answer_text: str
    start_position: int
    end_position: int


def main():
    args = parser.parse_args()

    logger.info(f"initialize model and converting weight from {args.model_path}")
    config = BertConfig.from_json(args.config_path)
    model = BertForSquad(config)
    load_tf_weight_to_pytorch_bert(model.bert, config, args.model_path)

    logger.info(f"initialize tokenizer using vocab {args.vocab_path}")
    tokenizer = SubWordTokenizer(args.vocab_path)

    logger.info(f"read squad dataset from {args.squad_train_path}")
    tokenizer = SubWordTokenizer(args.vocab_path)
    with open(args.squad_train_path, "r") as f:
        paragraphs = [paragraph for data in json.load(f)["data"] for paragraph in data["paragraphs"]]

    logger.info(f"convert squad dataset to features")
    examples: List[SquadExample] = []
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

    features = [
        convert_sequences_to_feature(
            tokenizer, (example.context_text, example.question_text), config.max_position_embeddings
        )
        for example in examples
    ]

    logger.info("create dataloader from squad features")
    input_ids = torch.tensor([feature.input_ids for feature in features])
    input_type_ids = torch.tensor([feature.input_type_ids for feature in features])
    input_mask = torch.tensor([feature.input_mask for feature in features])
    start_positions = torch.tensor([example.start_position for example in examples])
    end_positions = torch.tensor([example.end_position for example in examples])

    dataset = TensorDataset(input_type_ids, input_ids, input_mask, start_positions, end_positions)
    sampler = RandomSampler(dataset)
    train_loader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size)

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    logger.info("start training")
    logger.info(f"epoch: {args.epoch}")
    logger.info(f"batch size: {args.batch_size}")
    logger.info(f"length of dataset: {len(sampler)}")
    logger.info(f"length of steps per epoch: {len(train_loader)}")
    logger.info(f"learningrate: {args.learning_rate}")
    logger.info(f"logging steps: {args.logging_step}")
    logger.info(f"eval steps: {args.eval_step}")

    for epoch_index in range(args.epoch):
        model.train()
        running_loss = 0.0
        for batch_index, batch in enumerate(train_loader):
            optimizer.zero_grad()

            output = model(batch[0], batch[1], batch[2])

            loss = criterion(output, batch)
            running_loss += loss.item()


if __name__ == "__main__":
    main()

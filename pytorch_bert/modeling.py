import json

import torch
from torch import nn
from torch.nn import functional as F


class BertConfig:
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 16,
        initializer_range: float = 0.0,
        **kwargs,  # unused
    ):
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.type_vocab_size = type_vocab_size
        self.vocab_size = vocab_size

    @staticmethod
    def from_json(path: str) -> "BertConfig":
        with open(path, "r") as f:
            file_content = json.load(f)

        return BertConfig(**file_content)


def init_bert_weight(init_range: float):
    def fn_to_apply(module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=init_range)

            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    return fn_to_apply


class Bert(nn.Module):
    def __init__(self, config: BertConfig):
        super(Bert, self).__init__()
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.segment_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.embedding_layer_norm = nn.LayerNorm(config.hidden_size)
        self.embedding_dropout = nn.Dropout(p=config.hidden_dropout_prob)

        self.bert_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_attention_heads,
                dim_feedforward=config.intermediate_size,
                dropout=config.attention_probs_dropout_prob,
                activation=config.hidden_act,
            ),
            num_layers=config.num_hidden_layers,
        )

        self.pooler_layer = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor, attention_mask: torch.Tensor):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)

        words_embeddings = self.token_embeddings(input_ids)
        token_type_embeddings = self.segment_embeddings(token_type_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.embedding_layer_norm(embeddings)
        embeddings = self.embedding_dropout(embeddings)

        encoder_outputs = self.bert_encoder(embeddings.permute(1, 0, 2), src_key_padding_mask=attention_mask)

        pooled_output = self.activation(self.pooler_layer(encoder_outputs[0, :, :]))

        return encoder_outputs, pooled_output


class PretrainingBert(nn.Module):
    def __init__(self, config: BertConfig):
        super(PretrainingBert, self).__init__()

        self.bert = Bert(config)
        self.mlm = BertMLM(config)
        self.nsp = BertNSP(config)

    def forward(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor, attention_mask: torch.Tensor):
        encoder_outputs, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        mlm_output = self.mlm(encoder_outputs)
        nsp_output = self.nsp(pooled_output)

        return encoder_outputs, pooled_output, mlm_output, nsp_output


class BertMLM(nn.Module):
    def __init__(self, config: BertConfig):
        super(BertMLM, self).__init__()

        self.transform = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_layer_norm = nn.LayerNorm(config.hidden_size)

        self.output_layer = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.output_bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.output_logit = nn.Softmax(dim=2)

    def forward(self, encoder_outputs: torch.Tensor) -> torch.Tensor:
        transformed = F.gelu(self.transform(encoder_outputs))
        transformed = self.transform_layer_norm(transformed)

        logits = self.output_layer(transformed)
        return self.output_logit(logits + self.output_bias)


class BertNSP(nn.Module):
    def __init__(self, config: BertConfig):
        super(BertNSP, self).__init__()

        self.nsp_layer = nn.Linear(config.hidden_size, 2)
        self.nsp_softmax = nn.Softmax(dim=-1)

    def forward(self, pooled_output: torch.Tensor) -> torch.Tensor:
        return self.nsp_softmax(self.nsp_layer(pooled_output))

import os
import re
from typing import Iterable, Optional

import numpy as np
import torch

from .modeling import Bert, BertMLM, BertNSP

try:
    import tensorflow as tf

    _is_tf_imported = True
except ImportError:
    _is_tf_imported = False


class WeightConverter:
    def __init__(self, bert_model_path: str, n_transformer_block: Optional[int] = None):
        if not _is_tf_imported:
            raise ImportError("cannot import tensorflow, please install tensorflow first")

        if not os.path.isfile(f"{bert_model_path}.index"):
            raise ValueError(f"cannot find model {bert_model_path}")

        self.bert_model_path = bert_model_path
        self.tf_variable_map = {name: shape for name, shape in tf.train.list_variables(bert_model_path)}

        if n_transformer_block is not None:
            self.n_transformer_block = n_transformer_block
        else:
            self.n_transformer_block = _get_number_of_transformer_block(self.tf_variable_map.keys())

    def load_bert(self, bert: Bert):
        self.load_embedding_layer(bert)
        for layer_num in range(self.n_transformer_block):
            self.load_transformer_block(bert.bert_encoder.layers[layer_num], layer_num)
        self.load_pooler_layer(bert)

    def load_embedding_layer(self, bert: Bert):
        base = "bert/embeddings"

        self._load_embedding(bert.token_embeddings, f"{base}/word_embeddings")
        self._load_embedding(bert.segment_embeddings, f"{base}/token_type_embeddings")
        self._load_embedding(bert.position_embeddings, f"{base}/position_embeddings")
        self._load_layer_norm(bert.embedding_layer_norm, f"{base}/LayerNorm")

    def load_transformer_block(self, encoder: torch.nn.TransformerEncoder, layer_num: int):
        base = f"bert/encoder/layer_{layer_num}"

        self._load_self_attention(encoder.self_attn, f"{base}/attention")
        self._load_layer_norm(encoder.norm1, f"{base}/attention/output/LayerNorm")
        self._load_layer_norm(encoder.norm2, f"{base}/output/LayerNorm")
        self._load_linear(encoder.self_attn.out_proj, f"{base}/attention/output/dense")
        self._load_linear(encoder.linear1, f"{base}/intermediate/dense")
        self._load_linear(encoder.linear2, f"{base}/output/dense")

    def load_pooler_layer(self, bert: Bert):
        base = "bert/pooler"

        self._load_linear(bert.pooler_layer, f"{base}/dense")

    def load_mlm_head(self, bert_mlm: BertMLM):
        self._load_linear(bert_mlm.transform, "cls/predictions/transform/dense", load_bias=False)
        self._load_layer_norm(bert_mlm.transform_layer_norm, "cls/predictions/transform/LayerNorm")
        self._load_raw(bert_mlm.output_layer.weight, "bert/embeddings/word_embeddings")
        self._load_raw(bert_mlm.output_bias, "cls/predictions/output_bias")

    def load_nsp_head(self, bert_nsp: BertNSP):
        base = "cls/seq_relationship"
        self._load_raw(bert_nsp.nsp_layer.weight, f"{base}/output_weights")
        self._load_raw(bert_nsp.nsp_layer.bias, f"{base}/output_bias")

    def _load_raw(self, param, path):
        w = _load_tf_variable(self.bert_model_path, path)
        _load_torch_weight(param, w)

    def _load_embedding(self, param, embedding_path):
        embedding_weight = _load_tf_variable(self.bert_model_path, embedding_path)

        _load_torch_weight(param.weight, embedding_weight)

    def _load_linear(self, param: torch.nn.Linear, linear_base: str, load_bias: bool= True):
        linear_weight = _load_tf_variable(self.bert_model_path, f"{linear_base}/kernel")
        linear_weight = np.transpose(linear_weight)
        _load_torch_weight(param.weight, linear_weight)

        if load_bias:
            linear_bias = _load_tf_variable(self.bert_model_path, f"{linear_base}/bias")
            _load_torch_weight(param.bias, linear_bias)

    def _load_layer_norm(self, param: torch.nn.LayerNorm, layer_norm_base: str):
        layer_norm_gamma = _load_tf_variable(self.bert_model_path, f"{layer_norm_base}/gamma")
        layer_norm_beta = _load_tf_variable(self.bert_model_path, f"{layer_norm_base}/beta")

        _load_torch_weight(param.weight, layer_norm_gamma)
        _load_torch_weight(param.bias, layer_norm_beta)

    def _load_self_attention(self, param: torch.nn.MultiheadAttention, attention_base: str):
        query_weight = _load_tf_variable(self.bert_model_path, f"{attention_base}/self/query/kernel")
        key_weight = _load_tf_variable(self.bert_model_path, f"{attention_base}/self/key/kernel")
        value_weight = _load_tf_variable(self.bert_model_path, f"{attention_base}/self/value/kernel")

        query_weight = np.transpose(query_weight)
        key_weight = np.transpose(key_weight)
        value_weight = np.transpose(value_weight)

        query_bias = _load_tf_variable(self.bert_model_path, f"{attention_base}/self/query/bias")
        key_bias = _load_tf_variable(self.bert_model_path, f"{attention_base}/self/key/bias")
        value_bias = _load_tf_variable(self.bert_model_path, f"{attention_base}/self/value/bias")

        in_proj_weight = np.concatenate((query_weight, key_weight, value_weight))
        in_proj_bias = np.concatenate((query_bias, key_bias, value_bias))

        _load_torch_weight(param.in_proj_weight, in_proj_weight)
        _load_torch_weight(param.in_proj_bias, in_proj_bias)

def _get_number_of_transformer_block(tf_variable_names: Iterable[str]) -> int:
    layer_num_pattern = re.compile(r"bert/encoder/layer_([0-9]+)/")
    max_layer_num = 0
    for name in tf_variable_names:
        if not name.startswith("bert/encoder/layer_"):
            continue

        found = layer_num_pattern.search(name)
        if found:
            layer_num = int(found.group(1)) + 1
            max_layer_num = max(max_layer_num, layer_num)

    return max_layer_num


def _load_tf_variable(model_path: str, key: str):
    return tf.train.load_variable(model_path, key).squeeze()


def _load_torch_weight(param: torch.Tensor, data):
    assert param.shape == data.shape
    param.data = torch.from_numpy(data)

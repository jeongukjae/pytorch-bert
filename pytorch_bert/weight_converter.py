import numpy as np
import torch
from torch import nn

from .modeling import Bert, BertConfig, PretrainingBert

try:
    import tensorflow as tf

    _is_tf_imported = True
except ImportError:
    _is_tf_imported = False


def load_tf_weight_to_pytorch_bert(bert: Bert, config: BertConfig, tf_model_path: str):
    if not _is_tf_imported:
        raise ImportError("cannot import tensorflow, please install tensorflow first")

    # load embedding layer
    _load_embedding(bert.token_embeddings, tf_model_path, "bert/embeddings/word_embeddings")
    _load_embedding(bert.segment_embeddings, tf_model_path, "bert/embeddings/token_type_embeddings")
    _load_embedding(bert.position_embeddings, tf_model_path, "bert/embeddings/position_embeddings")
    _load_layer_norm(bert.embedding_layer_norm, tf_model_path, "bert/embeddings/LayerNorm")

    # load transformer encoders
    for layer_num in range(config.num_hidden_layers):
        encoder = bert.bert_encoder.layers[layer_num]
        encoder_path = f"bert/encoder/layer_{layer_num}"

        _load_self_attention(encoder.self_attn, tf_model_path, f"{encoder_path}/attention")
        _load_layer_norm(encoder.norm1, tf_model_path, f"{encoder_path}/attention/output/LayerNorm")
        _load_layer_norm(encoder.norm2, tf_model_path, f"{encoder_path}/output/LayerNorm")

        _load_linear(encoder.self_attn.out_proj, tf_model_path, f"{encoder_path}/attention/output/dense")
        _load_linear(encoder.linear1, tf_model_path, f"{encoder_path}/intermediate/dense")
        _load_linear(encoder.linear2, tf_model_path, f"{encoder_path}/output/dense")

    # load pooler layer
    _load_linear(bert.pooler_layer, tf_model_path, f"bert/pooler/dense")


def load_tf_weight_to_pytorch_pretraining_bert(
    bert: PretrainingBert, config: BertConfig, tf_model_path: str, share_parameters: bool = False
):
    load_tf_weight_to_pytorch_bert(bert.bert, config, tf_model_path)

    # load mlm
    _load_linear(bert.mlm.transform, tf_model_path, "cls/predictions/transform/dense", load_bias=False)
    _load_layer_norm(bert.mlm.transform_layer_norm, tf_model_path, "cls/predictions/transform/LayerNorm")
    if share_parameters:
        bert.mlm.output_layer.weight = bert.bert.token_embeddings.weight
    else:
        bert.mlm.output_layer.weight = nn.Parameter(bert.bert.token_embeddings.weight.clone())
    _load_raw(bert.mlm.output_bias, tf_model_path, "cls/predictions/output_bias")

    # load nsp
    _load_raw(bert.nsp.nsp_layer.weight, tf_model_path, f"cls/seq_relationship/output_weights")
    _load_raw(bert.nsp.nsp_layer.bias, tf_model_path, f"cls/seq_relationship/output_bias")


def _load_embedding(embedding: nn.Embedding, tf_model_path: str, embedding_path: str):
    embedding_weight = _load_tf_variable(tf_model_path, embedding_path)
    _load_torch_weight(embedding.weight, embedding_weight)


def _load_layer_norm(layer_norm: torch.nn.LayerNorm, tf_model_path: str, layer_norm_base: str):
    layer_norm_gamma = _load_tf_variable(tf_model_path, f"{layer_norm_base}/gamma")
    layer_norm_beta = _load_tf_variable(tf_model_path, f"{layer_norm_base}/beta")

    _load_torch_weight(layer_norm.weight, layer_norm_gamma)
    _load_torch_weight(layer_norm.bias, layer_norm_beta)


def _load_linear(linear: torch.nn.Linear, tf_model_path: str, linear_path: str, load_bias: bool = True):
    linear_weight = _load_tf_variable(tf_model_path, f"{linear_path}/kernel")
    linear_weight = np.transpose(linear_weight)
    _load_torch_weight(linear.weight, linear_weight)

    if load_bias:
        linear_bias = _load_tf_variable(tf_model_path, f"{linear_path}/bias")
        _load_torch_weight(linear.bias, linear_bias)


def _load_self_attention(param: torch.nn.MultiheadAttention, tf_model_path: str, attention_path: str):
    query_weight = _load_tf_variable(tf_model_path, f"{attention_path}/self/query/kernel")
    key_weight = _load_tf_variable(tf_model_path, f"{attention_path}/self/key/kernel")
    value_weight = _load_tf_variable(tf_model_path, f"{attention_path}/self/value/kernel")

    query_weight = np.transpose(query_weight)
    key_weight = np.transpose(key_weight)
    value_weight = np.transpose(value_weight)

    query_bias = _load_tf_variable(tf_model_path, f"{attention_path}/self/query/bias")
    key_bias = _load_tf_variable(tf_model_path, f"{attention_path}/self/key/bias")
    value_bias = _load_tf_variable(tf_model_path, f"{attention_path}/self/value/bias")

    in_proj_weight = np.concatenate((query_weight, key_weight, value_weight))
    in_proj_bias = np.concatenate((query_bias, key_bias, value_bias))

    _load_torch_weight(param.in_proj_weight, in_proj_weight)
    _load_torch_weight(param.in_proj_bias, in_proj_bias)


def _load_raw(param: torch.Tensor, tf_model_path: str, path: str):
    w = _load_tf_variable(tf_model_path, path)
    _load_torch_weight(param, w)


def _load_tf_variable(model_path: str, key: str):
    return tf.train.load_variable(model_path, key).squeeze()


def _load_torch_weight(param: torch.Tensor, data):
    assert param.shape == data.shape
    param.data = torch.from_numpy(data)

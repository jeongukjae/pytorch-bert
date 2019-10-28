# for testing
import os
import re

import numpy as np
import tensorflow as tf
import torch

from .modeling import Bert, BertNSP, BertMLM, BertConfig

config = BertConfig.from_json("./data/bert_config.json")
model = Bert(config)
nsp_head = BertNSP(config)
mlm_head = BertMLM(config)

tf_path = os.path.abspath("./data/bert_model.ckpt")
tf_variable_list = tf.train.list_variables(tf_path)

tf_variables = {}
num_encoder_layer = 0
layer_num_pattern = re.compile(r"bert/encoder/layer_([0-9]+)/")

for name, shape in tf_variable_list:
    print(f"Loading Tensorflow weight {name} with shape {shape}")
    variable_data = tf.train.load_variable(tf_path, name)

    tf_variables[name] = variable_data.squeeze()
    if "bert/encoder/layer_" in name:
        found = layer_num_pattern.search(name)
        if found:
            num_encoder_layer = max(int(found.group(1)) + 1, num_encoder_layer)

print(f"total number of encoder layer: {num_encoder_layer}")

embedding_layer_mapping = {
    "embedding_layer_norm": {"weight": "bert/embeddings/LayerNorm/gamma", "bias": "bert/embeddings/LayerNorm/beta"},
    "token_embeddings": {"weight": "bert/embeddings/word_embeddings"},
    "segment_embeddings": {"weight": "bert/embeddings/token_type_embeddings"},
    "position_embeddings": {"weight": "bert/embeddings/position_embeddings"},
}
pooler_mapping = {"pooler_layer": {"weight": "bert/pooler/dense/kernel", "bias": "bert/pooler/dense/bias"}}


def set_data_to_model(current_layer, map_dict, current_path, tf_variables):
    for key in map_dict.keys():
        if isinstance(map_dict[key], dict):
            set_data_to_model(getattr(current_layer, key), map_dict[key], f"{current_path}.{key}", tf_variables)
        else:
            param_to_set = getattr(current_layer, key)
            path_to_set = f"{current_path}.{key}"

            data = tf_variables[map_dict[key]]
            if "dense/weight" in map_dict[key]:
                data = np.transpose(data)

            print(f"load {path_to_set} from {map_dict[key]}")

            set_param_data(param_to_set, data, f"Cannot load weight {path_to_set} from {map_dict[key]}")


def set_param_data(param, data, err_msg="Cannot load weight"):
    assert param.shape == data.shape, err_msg
    param.data = torch.from_numpy(data)


def set_encoder_data_to_model(model, tf_variables, num_encoder_layer):
    for i in range(num_encoder_layer):
        base_path = f"bert/encoder/layer_{i}"
        encoder_layer = model.bert_encoder.layers[i]
        print(f"load .bert_encoder.layers.{i} from {base_path}")

        qk = np.transpose(tf_variables[f"{base_path}/attention/self/query/kernel"])
        qb = tf_variables[f"{base_path}/attention/self/query/bias"]
        kk = np.transpose(tf_variables[f"{base_path}/attention/self/key/kernel"])
        kb = tf_variables[f"{base_path}/attention/self/key/bias"]
        vk = np.transpose(tf_variables[f"{base_path}/attention/self/value/kernel"])
        vb = tf_variables[f"{base_path}/attention/self/value/bias"]

        in_proj_weight = np.concatenate((qk, kk, vk))
        in_proj_bias = np.concatenate((qb, kb, vb))

        set_param_data(encoder_layer.self_attn.in_proj_weight, in_proj_weight)
        set_param_data(encoder_layer.self_attn.in_proj_bias, in_proj_bias)

        set_param_data(
            encoder_layer.self_attn.out_proj.weight,
            np.transpose(tf_variables[f"{base_path}/attention/output/dense/kernel"]),
        )
        set_param_data(encoder_layer.self_attn.out_proj.bias, tf_variables[f"{base_path}/attention/output/dense/bias"])
        set_param_data(encoder_layer.norm1.weight, tf_variables[f"{base_path}/attention/output/LayerNorm/gamma"])
        set_param_data(encoder_layer.norm1.bias, tf_variables[f"{base_path}/attention/output/LayerNorm/beta"])
        set_param_data(
            encoder_layer.linear1.weight, np.transpose(tf_variables[f"{base_path}/intermediate/dense/kernel"])
        )
        set_param_data(encoder_layer.linear1.bias, tf_variables[f"{base_path}/intermediate/dense/bias"])
        set_param_data(encoder_layer.linear2.weight, np.transpose(tf_variables[f"{base_path}/output/dense/kernel"]))
        set_param_data(encoder_layer.linear2.bias, tf_variables[f"{base_path}/output/dense/bias"])
        set_param_data(encoder_layer.norm2.weight, tf_variables[f"{base_path}/output/LayerNorm/gamma"])
        set_param_data(encoder_layer.norm2.bias, tf_variables[f"{base_path}/output/LayerNorm/beta"])


set_data_to_model(model, embedding_layer_mapping, "", tf_variables)
set_encoder_data_to_model(model, tf_variables, num_encoder_layer)
set_data_to_model(model, pooler_mapping, "", tf_variables)

nsp_head.nsp_layer.weight.data = torch.from_numpy(tf_variables["cls/seq_relationship/output_weights"])
nsp_head.nsp_layer.bias.data = torch.from_numpy(tf_variables["cls/seq_relationship/output_bias"])
mlm_head.transform.weight.data = torch.from_numpy(np.transpose(tf_variables["cls/predictions/transform/dense/kernel"]))
mlm_head.transform_layer_norm.weight.data = torch.from_numpy(tf_variables["cls/predictions/transform/LayerNorm/gamma"])
mlm_head.transform_layer_norm.bias.data = torch.from_numpy(tf_variables["cls/predictions/transform/LayerNorm/beta"])
mlm_head.output_layer.weight.data = torch.from_numpy(tf_variables["bert/embeddings/word_embeddings"])
mlm_head.output_bias.data = torch.from_numpy(tf_variables["cls/predictions/output_bias"])

model.eval()
nsp_head.eval()
mlm_head.eval()

# torch.save(model.state_dict(), "./data/bert_model_torch.pth")
# import torch

from .bert import FeatureExtractor
from .tokenizer import SubWordTokenizer

# from .modeling import Bert, BertConfig

tokenizer = SubWordTokenizer("./data/vocab.txt")
# bert = Bert(BertConfig.from_json("./data/bert_config.json"))
# bert.load_state_dict(torch.load("./data/bert_model_torch.pth"))
# bert.eval()
feature = FeatureExtractor(tokenizer)

sequences = (
    "One use case for this notation is that it allows pure Python functions to fully emulate behaviors of existing C coded functions.",
    "For example, the built-in pow() function does not accept keyword arguments",
)

tokens, input_type_ids, input_ids, input_mask = feature.convert_sequences_to_feature(sequences, 512)
pos = (3, 5, 8, 12, 19, 35)
answer = []
for p in pos:
    answer.append(tokens[p])
    tokens[p] = "[MASK]"
    input_ids[p] = 103

# tokens, input_type_ids, input_ids, input_mask = feature.convert_sequences_to_feature(
#     ("the man went to the store .", "penguins are flightless ."), 128
# )

print("sequences : ", sequences)
print("tokens : ", tokens)
# print(input_type_ids)
print("input_ids : ", input_ids)
# print(input_mask)


mask = torch.ones((1, 512), dtype=torch.bool) ^ torch.tensor(input_mask, dtype=torch.bool).unsqueeze(0)

encoder_outputs, pooled_output = model(torch.tensor([input_ids]), torch.tensor([input_type_ids]), mask)
# print(encoder_outputs.size())
# print(pooled_output)
# print(nsp_head(pooled_output))
mlm_output = mlm_head(encoder_outputs)

for p, a in zip(pos, answer):
    print(f"mlm output (pos {p}): {tokenizer.inv_vocab[torch.argmax(mlm_output[p,:,:], -1).item()]}, answer: {a}")

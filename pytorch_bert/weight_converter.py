import os
from pytorch_bert.modeling import Bert, BertMLM, BertNSP

try:
    import tensorflow as tf

    _is_tf_imported = True
except ImportError:
    _is_tf_imported = False


class WeightConverter:
    def __init__(self, bert_model_path: str):
        if not _is_tf_imported:
            raise ImportError("cannot import tensorflow, please install tensorflow first")

        if not os.path.isfile(f"{bert_model_path}.index"):
            raise ValueError(f"cannot find model {bert_model_path}")

        self.bert_model_path = bert_model_path
        self.tf_variable_list = tf.train.list_variables(bert_model_path)

    def load_bert(self, bert: Bert):
        self.load_embedding_layer(bert)
        self.load_transformer_block(bert)

    def load_embedding_layer(self, bert: Bert):
        pass

    def load_transformer_block(self, bert: Bert):
        pass

    def load_mlm_head(self, bert_mlm: BertMLM):
        pass

    def load_nsp_head(self, bert_nsp: BertNSP):
        pass

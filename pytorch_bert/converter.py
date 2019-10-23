# for debugging now

import os
import tensorflow as tf

tf_path = os.path.abspath("./data/bert_model.ckpt")
tf_vars = tf.train.list_variables(tf_path)
print(*tf_vars, sep="\n")

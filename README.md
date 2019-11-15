# pytorch-bert

[![codecov](https://codecov.io/gh/jeongukjae/pytorch-bert/branch/master/graph/badge.svg)](https://codecov.io/gh/jeongukjae/pytorch-bert)
[![CircleCI](https://circleci.com/gh/jeongukjae/pytorch-bert.svg?style=shield)](https://circleci.com/gh/jeongukjae/pytorch-bert)
[![PyPI](https://img.shields.io/pypi/v/pytorch-bert)](https://pypi.org/project/pytorch-bert/)

A implementation of BERT using PyTorch `TransformerEncoder` and pre-trained model of [google-research/bert](https://github.com/google-research/bert).

## Installation

```sh
pip install pytorch-bert
```

## Usage

- [**Usage**](https://github.com/jeongukjae/pytorch-bert/blob/8c276c222e721bc725049599f6b46dfedbc63340/tests/test_converter.py#L32)
- [**Input, Ouptut Shape**](https://github.com/jeongukjae/pytorch-bert/blob/master/tests/test_modeling.py)

```python
config = BertConfig.from_json("path-to-pretarined-weights/bert_config.json")
model = Bert(config)
load_tf_weight_to_pytorch_bert(model, config, "path-to-pretarined-weights/bert_model.ckpt")
```

Download model files in [google-research/bert](https://github.com/google-research/bert) repository.

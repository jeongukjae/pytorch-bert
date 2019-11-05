import os
import urllib.request
import zipfile

import pytest

from pytorch_bert.modeling import Bert, BertConfig, PretrainingBert
from pytorch_bert.weight_converter import load_tf_weight_to_pytorch_bert, load_tf_weight_to_pytorch_pretraining_bert

google_bert_model_parameters = pytest.mark.parametrize(
    "url,directory,unzipped_path",
    [
        pytest.param(
            "https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip",
            "/tmp/bert-base",
            "/tmp/bert-base/multi_cased_L-12_H-768_A-12",
        ),
        pytest.param(
            "https://storage.googleapis.com/bert_models/2019_05_30/wwm_uncased_L-24_H-1024_A-16.zip",
            "/tmp/bert-large",
            "/tmp/bert-large/wwm_uncased_L-24_H-1024_A-16",
        ),
    ],
)


@google_bert_model_parameters
def test_convert_pretrained_weight_bert(url: str, directory: str, unzipped_path: str):
    if not os.path.isdir(directory):
        os.mkdir(directory)

    download_model_file(url, directory)

    config = BertConfig.from_json(f"{unzipped_path}/bert_config.json")
    model = Bert(config)
    load_tf_weight_to_pytorch_bert(model, config, f"{unzipped_path}/bert_model.ckpt")


@google_bert_model_parameters
def test_convert_pretrained_weight_of_pretraining_bert(url: str, directory: str, unzipped_path: str):
    if not os.path.isdir(directory):
        os.mkdir(directory)

    download_model_file(url, directory)

    config = BertConfig.from_json(f"{unzipped_path}/bert_config.json")
    model = PretrainingBert(config)
    load_tf_weight_to_pytorch_pretraining_bert(model, config, f"{unzipped_path}/bert_model.ckpt")

    assert id(model.bert.token_embeddings.weight) != id(model.mlm.output_layer.weight)


@google_bert_model_parameters
def test_convert_pretrained_weight_of_pretraining_bert_sharing_parameters(url: str, directory: str, unzipped_path: str):
    if not os.path.isdir(directory):
        os.mkdir(directory)

    download_model_file(url, directory)

    config = BertConfig.from_json(f"{unzipped_path}/bert_config.json")
    model = PretrainingBert(config)
    load_tf_weight_to_pytorch_pretraining_bert(model, config, f"{unzipped_path}/bert_model.ckpt", share_parameters=True)

    assert id(model.bert.token_embeddings.weight) == id(model.mlm.output_layer.weight)


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

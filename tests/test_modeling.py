import torch
import pytest

import pytorch_bert.modeling as M


@pytest.fixture
def config():
    return M.BertConfig(300, num_hidden_layers=3)


def test_bert_with_random_input(config: M.BertConfig):
    model = M.Bert(config)
    model.apply(M.init_bert_weight(config.initializer_range))
    batch_size = 1

    encoder_ouputs, pooled_output = model(
        torch.randint(300, (batch_size, config.max_position_embeddings)),
        torch.randint(2, (batch_size, config.max_position_embeddings)),
        torch.randint(2, (batch_size, config.max_position_embeddings), dtype=torch.bool),
    )

    assert encoder_ouputs.size() == (config.max_position_embeddings, batch_size, config.hidden_size)
    assert pooled_output.size() == (batch_size, config.hidden_size)


def test_pretraining_bert_with_random_input(config: M.BertConfig):
    model = M.PretrainingBert(config)
    model.apply(M.init_bert_weight(config.initializer_range))
    batch_size = 1

    encoder_ouputs, pooled_output, mlm_output, nsp_output = model(
        torch.randint(300, (batch_size, config.max_position_embeddings)),
        torch.randint(2, (batch_size, config.max_position_embeddings)),
        torch.randint(2, (batch_size, config.max_position_embeddings), dtype=torch.bool),
    )

    assert encoder_ouputs.size() == (config.max_position_embeddings, batch_size, config.hidden_size)
    assert pooled_output.size() == (batch_size, config.hidden_size)
    assert mlm_output.size() == (config.max_position_embeddings, batch_size, config.vocab_size)
    assert nsp_output.size() == (batch_size, 2)

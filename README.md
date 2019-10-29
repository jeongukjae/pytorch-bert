# pytorch-bert

A BERT implementation written in PyTorch with pre-trained model from [google-research/bert](https://github.com/google-research/bert)

## Usage

### Convert Weights

```sh
$ pytorch_bert convert \
    -m ./data/bert_model.ckpt \
    -c ./data/bert_config.json \
    -o ./data/ \
    --load-masked-lm-head \
    --load-nsp-head
```

### Test Masked LM

```sh
$ pytorch_bert masked-lm \
    "One use case for this notation is that it allows pure Python functions to fully emulate behaviors of existing C coded functions." \
    "For example, the built-in pow() function does not accept keyword arguments" \
    -c ./data/bert_config.json \
    -m ./data/bert_model_torch.pth \
    --head ./data/bert_model_torch_mlm.pth \
    -v ./data/vocab.txt

input text: ('One use case for this notation is that it allows pure Python functions to fully emulate behaviors of existing C coded functions.', 'For example, the built-in pow() function does not accept keyword arguments')
parsed tokens: ['[CLS]', 'one', 'use', 'case', 'for', 'this', 'notation', 'is', 'that', 'it', 'allows', 'pure', 'python', '[MASK]', 'to', 'fully', 'em', '##ulate', 'behaviors', 'of', 'existing', 'c', 'coded', 'functions', '.', '[SEP]', 'for', '[MASK]', ',', 'the', 'built', '-', 'in', 'pow', '[MASK]', ')', 'function', '[MASK]', 'not', 'accept', 'key', '[MASK]', 'arguments', '[SEP]']

masked pos 13. answer: functions, predicted: functions
masked pos 27. answer: example, predicted: example
masked pos 34. answer: (, predicted: (
masked pos 37. answer: does, predicted: does
masked pos 41. answer: ##word, predicted: ##word
```

### Test NSP

```sh
$ pytorch_bert nsp \
    "One use case for this notation is that it allows pure Python functions to fully emulate behaviors of existing C coded functions." \
    "For example, the built-in pow() function does not accept keyword arguments" \
    -c ./data/bert_config.json \
    -m ./data/bert_model_torch.pth \
    --head ./data/bert_model_torch_nsp.pth \
    -v ./data/vocab.txt

input text: ('One use case for this notation is that it allows pure Python functions to fully emulate behaviors of existing C coded functions.', 'For example, the built-in pow() function does not accept keyword arguments')
parsed tokens: ['[CLS]', 'one', '[MASK]', '[MASK]', 'for', 'this', 'notation', 'is', 'that', '[MASK]', 'allows', '[MASK]', '[MASK]', 'functions', 'to', 'fully', 'em', '##ulate', 'behaviors', 'of', 'existing', 'c', '[MASK]', 'functions', '.', '[SEP]', 'for', 'example', ',', 'the', 'built', '-', 'in', 'pow', '(', ')', 'function', 'does', 'not', 'accept', 'key', '##word', 'arguments', '[SEP]']

nsp : IsNext
```

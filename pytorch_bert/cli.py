import os
from argparse import ArgumentParser

import torch

from .feature import FeatureExtractor
from .modeling import Bert, BertConfig, BertMLM, BertNSP
from .tokenizer import SubWordTokenizer, SpecialToken, Vocab
from .weight_converter import WeightConverter

parser = ArgumentParser("bert")
subparsers = parser.add_subparsers(help='Commands', dest='command')
convert_parser = subparsers.add_parser('convert', help ='A command that converts tf weights to pytorch weight. This command needs tensorflow.')
convert_parser.add_argument('-c', '--config', required=True, dest='config_path', help='A configuration file path of pretrained model')
convert_parser.add_argument('-m', '--model', required=True, dest='model_path', help='A pretrained model path of tensorflow')
convert_parser.add_argument('-o', '--out', required=True, dest='output_folder', help="output folder of pytorch weight")
convert_parser.add_argument('-f', '--full', action='store_true', default=False, help='save pytorch weight with full model (default: save only state-dict)')
convert_parser.add_argument('--load-masked-lm-head', action='store_true', default=False, help='Enable loading Maksed LM head')
convert_parser.add_argument('--load-nsp-head', action='store_true', default=False, help='Enable loading NSP head')

masked_lm_parser = subparsers.add_parser('masked-lm', help ='A command that tests mask lm using pretrained model')
masked_lm_parser.add_argument('first', help='A first one of sequence pair')
masked_lm_parser.add_argument('second', nargs="?", help='A second one of sequence pair (optional)')
masked_lm_parser.add_argument('-c', '--config', required=True, dest='config_path', help='A configuration file path of pretrained model')
masked_lm_parser.add_argument('-v', '--vocab', required=True, dest='vocab_path', help='A vocab file path')
masked_lm_parser.add_argument('-m', '--model', required=True, dest='model_path', help='A converted model that contains Bert weight')
masked_lm_parser.add_argument('--head', required=True, dest='masked_lm_head_path', help='A converted model that contains Masked LM Head weight')

nsp_parser = subparsers.add_parser('nsp', help ='A command that tests mask lm using pretrained model')
nsp_parser.add_argument('first', help='A first one of sequence pair')
nsp_parser.add_argument('second', help='A second one of sequence pair')
nsp_parser.add_argument('-c', '--config', required=True, dest='config_path', help='A configuration file path of pretrained model')
nsp_parser.add_argument('-v', '--vocab', required=True, dest='vocab_path', help='A vocab file path')
nsp_parser.add_argument('-m', '--model', required=True, dest='model_path', help='A converted model that contains Bert weight')
nsp_parser.add_argument('--head', required=True, dest='nsp_head_path', help='A converted model that contains NSP Head weight')


def main():
    args = parser.parse_args()

    if args.command == "convert":
        _convert_tf_weight_to_pytorch(args)
    elif args.command == "masked-lm":
        _test_masked_lm(args)
    elif args.command == "nsp":
        _test_nsp(args)


def _convert_tf_weight_to_pytorch(args):
    if not os.path.exists(args.output_folder):
        os.mkdir(args.output_folder)
    elif os.path.isfile(args.output_folder):
        raise ValueError(f"{args.output_folder} is a file. You have to pass a folder name to this flag.")

    config = BertConfig.from_json(args.config_path)
    bert = Bert(config)

    weight_converter = WeightConverter(args.model_path)
    weight_converter.load_bert(bert)
    _save_model(bert, os.path.join(args.output_folder, 'bert_model_torch.pth'), args.full)

    if args.load_masked_lm_head:
        mlm_head = BertMLM(config)
        weight_converter.load_mlm_head(mlm_head)
        _save_model(mlm_head, os.path.join(args.output_folder, 'bert_model_torch_mlm.pth'), args.full)

    if args.load_nsp_head:
        nsp_head = BertNSP(config)
        weight_converter.load_nsp_head(nsp_head)
        _save_model(nsp_head, os.path.join(args.output_folder, 'bert_model_torch_nsp.pth'), args.full)


def _test_masked_lm(args):
    config = BertConfig.from_json(args.config_path)

    bert = _load_model(Bert(config), args.model_path)
    mlm_head = _load_model(BertMLM(config), args.masked_lm_head_path)

    vocab = Vocab(args.vocab_path)
    tokenizer = SubWordTokenizer(vocab)
    feature_extractor = FeatureExtractor(tokenizer)

    sequence_pair = (args.first,)
    if args.second:
        sequence_pair += (args.second,)

    feature = feature_extractor.convert_sequences_to_feature(sequence_pair, config.max_position_embeddings)
    feature, masked_labels = feature_extractor.mask(feature, vocab.vocab[SpecialToken.mask])

    input_ids = torch.tensor([feature.input_ids])
    input_type_ids = torch.tensor([feature.input_type_ids])
    input_mask = feature_extractor.create_input_mask(feature.input_mask, config.max_position_embeddings)

    print("")
    print(f'input text: {sequence_pair}')
    print(f"parsed tokens: {feature.tokens}")
    print("")

    encoder_outputs, _ = bert(input_ids, input_type_ids, input_mask)
    mlm_output = mlm_head(encoder_outputs)

    for index, pos in enumerate(masked_labels.positions):
        predicted = torch.argmax(mlm_output[pos,:,:], -1).item()
        predicted = vocab.inv_vocab[predicted]
        answer = vocab.inv_vocab[masked_labels.answers[index]]
        print(f'masked pos {pos}. answer: {answer}, predicted: {predicted}')


def _test_nsp(args):
    config = BertConfig.from_json(args.config_path)

    bert = _load_model(Bert(config), args.model_path)
    nsp_head = _load_model(BertNSP(config), args.nsp_head_path)

    vocab = Vocab(args.vocab_path)
    tokenizer = SubWordTokenizer(vocab)
    feature_extractor = FeatureExtractor(tokenizer)

    sequence_pair = (args.first,args.second)

    feature = feature_extractor.convert_sequences_to_feature(sequence_pair, config.max_position_embeddings)
    feature, _ = feature_extractor.mask(feature, vocab.vocab[SpecialToken.mask])

    input_ids = torch.tensor([feature.input_ids])
    input_type_ids = torch.tensor([feature.input_type_ids])
    input_mask = feature_extractor.create_input_mask(feature.input_mask, config.max_position_embeddings)

    print("")
    print(f'input text: {sequence_pair}')
    print(f"parsed tokens: {feature.tokens}")
    print("")

    _, pooled_output = bert(input_ids, input_type_ids, input_mask)
    nsp_output = nsp_head(pooled_output)

    print(f'nsp : {"IsNext" if torch.argmax(nsp_output).item() == 0 else "NotNext"}')



def _save_model(model: torch.nn.Module, path: str, is_full: bool):
    if is_full:
        torch.save(model, path)
    else:
        torch.save(model.state_dict(), path)

def _load_model(model: torch.nn.Module, path: str, is_full: bool = False):
    if is_full:
        model = torch.load(path)
        model.eval()
    else:
        model.load_state_dict(torch.load(path))
        model.eval()

    return model

import os
import torch
from argparse import ArgumentParser
from .modeling import Bert, BertConfig, BertMLM, BertNSP
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

nsp_parser = subparsers.add_parser('nsp', help ='A command that tests mask lm using pretrained model')


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
    pass


def _test_nsp(args):
    pass

def _save_model(model: torch.nn.Module, path: str, is_full: bool):
    if is_full:
        torch.save(model, path)
    else:
        torch.save(model.state_dict(), path)

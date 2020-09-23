import sys

sys.path += ['.']
sys.path += ['..']

import os
import os.path as osp
import argparse
from dnn.runners.build_evaluator import build_evaluator
from dnn.typing_definitions import *
from dnn.data_utils.build_dataloader import build_dataloader
import json


def fill_parser(parser):
    parser.add_argument('--exp-dir', dest='experiment_dir', help='path for experiment dir',
                        default=None, type=str)
    parser.add_argument('--tabolar-mode', dest='is_tabolar_mode', help='yes for tabolar mode', choices=['yes', 'no'],
                        default='no', type=str)
    parser.add_argument('--cpugpu', default='gpu', choices=['cpu', 'gpu'],
                        help='where to run the evaluation (e.g., for debugging)')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    return parser


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Arguments for Coronavirus network")
    fill_parser(parser)

    args = parser.parse_args()
    return args


def main(experiment_dir: DirPath, data_params: dict, net_params: dict, eval_params: dict, data_set: str, gpu_list: list,
         is_tabolar_mode, cpugpu: str, tabolar_data_path):

    # Set GPU
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_list)[1:-1]

    # Build and load data set
    test_loader = build_dataloader(data_params=data_params, data_set=data_set, tabolar_data_path=tabolar_data_path,
                                   is_tabolar_mode=is_tabolar_mode)

    # Build an evaluator
    evaluator = build_evaluator(experiment_dir=experiment_dir, net_params=net_params, image_size=data_params['image_size'],
                                is_tabolar_mode=is_tabolar_mode, weight_path=eval_params['weights_path'],
                                data_set=data_set, cpugpu=cpugpu)

    evaluator.run(test_loader)


def run(args):
    # Initialize config
    config_path = osp.join(args.experiment_dir, "config.json")

    with open(config_path, 'r') as f:
        params = json.load(f)

    # Initialize params
    conf_data_params = params['data_params']
    conf_net_params = params['net_params']
    conf_eval_params = params['eval_params']
    conf_gpu_list = params['gpu_list']


    # Eval the model
    main(experiment_dir=args.experiment_dir, data_params=conf_data_params, net_params=conf_net_params,
         eval_params=conf_eval_params, data_set='test', gpu_list=conf_gpu_list, cpugpu=args.cpugpu,
         is_tabolar_mode=args.is_tabolar_mode, tabolar_data_path=conf_data_params["tabolar_data_path"])


if __name__ == '__main__':
    args = parse_args()
    run(args)
    print("finish")

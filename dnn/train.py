import argparse
import json
import os
import argparse
import os.path as osp
import sys
from dnn.typing_definitions import *
from dnn.data_utils.build_dataloader import build_dataloader
from dnn.runners.build_trainer import build_trainer


def fill_parser(parser):
    parser.add_argument('--runmode', choices=['new', 'resume'], default='new',
                        help="new or resume training - resume works only on new runs")
    parser.add_argument('--exp-dir', dest='experiment_dir', help='path for experiment dir',
                        default=None, type=str)
    parser.add_argument('--tabolar-mode', dest='is_tabolar_mode', help='yes for tabolar mode', choices=['yes', 'no'],
                        default='no', type=str)
    parser.add_argument('--cpugpu', default='gpu', choices=['cpu', 'gpu'],
                        help='where to run the training (e.g., for debugging)')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Arguments for Coronavirus network")
    fill_parser(parser)

    args = parser.parse_args()
    return args


def main(experiment_dir: DirPath, data_params: dict, net_params: dict, train_params: dict, gpu_list: list,
         optimizer_params: dict, tabolar_data_path: FilePath, is_tabolar_mode: str, runmode: str, cpugpu:str = 'gpu'):

    # Set GPU
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_list)[1:-1]

    # Build and load data set
    train_loader = build_dataloader(data_params=data_params, data_set='train', tabolar_data_path=tabolar_data_path,
                                    is_tabolar_mode=is_tabolar_mode)
    test_loader = build_dataloader(data_params=data_params, data_set='test', tabolar_data_path=tabolar_data_path,
                                   is_tabolar_mode=is_tabolar_mode)

    # Build a trainer
    trainer = build_trainer(experiment_dir=experiment_dir, train_data_loader=train_loader, test_data_loader=test_loader,
                            train_params=train_params, net_params=net_params, image_size=data_params['image_size'],
                            optimizer_params=optimizer_params, runmode=runmode, is_tabolar_mode=is_tabolar_mode,
                            cpugpu=cpugpu)

    trainer.run(train_loader, max_epochs=train_params['max_epochs'])


def run(args):

    config_path = osp.join(args.experiment_dir, "config.json")
    config_path = osp.join(args.experiment_dir, "config.json")

    with open(config_path, 'r') as f:
        params = json.load(f)

    # Initialize params
    conf_data_params = params['data_params']
    conf_net_params = params['net_params']
    conf_train_params = params['train_params']
    conf_gpu_list = params['gpu_list']
    conf_optimizer_params = params['train_params'][params['train_params']['optimizer_type']]

    # Train the model
    main(experiment_dir=args.experiment_dir, data_params=conf_data_params, net_params=conf_net_params,
         train_params=conf_train_params, gpu_list=conf_gpu_list, optimizer_params=conf_optimizer_params,
         tabolar_data_path=conf_data_params["tabolar_data_path"], runmode=args.runmode, cpugpu=args.cpugpu,
         is_tabolar_mode=args.is_tabolar_mode)


if __name__ == '__main__':
    # Initialize config
    args = parse_args()
    run(args)
    print("finish")

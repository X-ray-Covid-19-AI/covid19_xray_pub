from dnn.typing_definitions import *
from dnn.net.feclinic_net import FeClinicNet
from dnn.net.model_factory import ModelFactory
from dnn.runners.builder_utils import build_metrics
# from ignite.engine import create_supervised_evaluator
from dnn.runners.builder_utils import choose_prepare_batch
from dnn.runners.create_supervised_runners_custom import create_supervised_evaluator
import torch
import collections
import os
import os.path as osp
import torch.utils.data as data
from dnn.runners.training_events import attach_evaluator_events
import torch.nn as nn


def create_logging_dir_for_eval(log_dir):

    eval_dir = osp.join(log_dir, 'eval')
    os.makedirs(eval_dir, exist_ok=True)

    return eval_dir


def build_evaluator(experiment_dir: DirPath, net_params: dict, image_size: int, weight_path: FilePath, data_set: str,
                    is_tabolar_mode: str, cpugpu: str):

    if is_tabolar_mode == 'yes':
        mfctr = FeClinicNet(net_params["net_name"], net_params[net_params['net_name']], net_params["classifier_fc_size"],
                            pretrained=False, image_size=image_size)
    else:
        mfctr = ModelFactory.create_model(net_params["net_name"], net_params[net_params['net_name']], pretrained=False,
                                          classifier_layer_size_list=net_params['classifier_layer_size'], image_size=image_size)

    if cpugpu == 'gpu':
        model = mfctr.cuda()
        device = 'cuda'
    else:
        model = mfctr.cpu()
        device = None

    checkpoint = torch.load(weight_path, map_location=torch.device('cpu'))
    if type(checkpoint) == collections.OrderedDict:  # in previous runs we saved checkpoints as OrderDict
        model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint['model'])
    # net = nn.DataParallel(net, device_ids=train_params['gpu_list'])  # parallel

    metrics = build_metrics()
    evaluator = create_supervised_evaluator(model=model, metrics=metrics, non_blocking=True, is_tabolar_mode=is_tabolar_mode,
                                            prepare_batch=choose_prepare_batch(is_tabolar_mode), device=device)

    attach_evaluator_events(evaluator=evaluator, experiment_dir=experiment_dir, data_set=data_set)

    return evaluator

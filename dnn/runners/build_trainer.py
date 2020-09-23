from dnn.typing_definitions import *
import ignite
import torch.utils.data as data
import os.path as osp
import os
from dnn.net.model_factory import ModelFactory
from dnn.net.feclinic_net import FeClinicNet
import torch.nn as nn
import shutil
from dnn.runners.build_evaluator import create_logging_dir_for_eval
from dnn.optimizer.optimizer_factory import OptimizerFactory
from dnn.runners.builder_utils import build_metrics
from dnn.runners.builder_utils import choose_prepare_batch
from dnn.runners.create_supervised_runners_custom import create_supervised_evaluator, create_supervised_trainer
# from ignite.engine import create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import ModelCheckpoint
from tensorboardX import SummaryWriter
import torch
from dnn.runners.training_events import attach_trainer_events


def get_last_checkpoint(checkpoint_dir: str):
    s = sorted([f for f in os.listdir(checkpoint_dir) if 'pth' in f],
               key=lambda f: int(f.split('.')[0].rsplit('_', 1)[-1]))
    return osp.join(checkpoint_dir, s[-1])


def build_trainer(experiment_dir: DirPath, train_data_loader: data.DataLoader, test_data_loader: data.DataLoader,
                  train_params: dict, net_params: dict, image_size: int, optimizer_params: dict,
                  is_tabolar_mode: str, runmode: str, cpugpu:str = 'gpu') -> ignite.engine:

    checkpoint_dir = osp.join(experiment_dir, train_params['checkpoint_relative_path'])
    logging_dir = osp.join(experiment_dir, train_params['logging_dir_relative_path'])  # the log
    tb_dir = osp.join(logging_dir, 'tensorboard')

    if is_tabolar_mode == 'yes':
        mfctr = FeClinicNet(net_params['net_name'], net_params[net_params['net_name']], net_params['classifier_fc_size'],
                            net_params['pretrained'], image_size)
    else:
        mfctr = ModelFactory.create_model(net_params['net_name'], net_params[net_params['net_name']],
                                          net_params['pretrained'], net_params['classifier_layer_size'], image_size)

    if cpugpu == 'gpu':
        model = mfctr.cuda()
        device = 'cuda'
    else:
        model = mfctr.cpu()
        device = None

    loss = nn.CrossEntropyLoss()
    optimizer = OptimizerFactory.create_optimizer(optimizer_name=train_params['optimizer_type'],
                                                  net_params=model.parameters(),
                                                  optimizer_params=optimizer_params)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9261)
    summary_writer = SummaryWriter(log_dir=tb_dir)

    trainer = create_supervised_trainer(model=model, optimizer=optimizer, device=device, is_tabolar_mode=is_tabolar_mode,
                                        prepare_batch=choose_prepare_batch(is_tabolar_mode), non_blocking=True, loss_fn=loss)
    metrics = build_metrics()
    evaluator = create_supervised_evaluator(model=model, metrics=metrics, device=device, non_blocking=True,
                                            prepare_batch=choose_prepare_batch(is_tabolar_mode), is_tabolar_mode=is_tabolar_mode)
    eval_dir = create_logging_dir_for_eval(log_dir=logging_dir)

    checkpoint_handler = ModelCheckpoint(dirname=checkpoint_dir,
                                         filename_prefix='checkpoint',
                                         save_interval=1,
                                         n_saved=30,
                                         atomic=True,
                                         require_empty=False,
                                         create_dir=True,
                                         save_as_state_dict=True)

    if runmode == 'new':
        shutil.rmtree(tb_dir, ignore_errors=True)
        starting_epoch = 0
    elif runmode == 'resume':
        to_load = {'trainer': trainer, 'model': model, 'optimizer': optimizer}
        checkpoint = torch.load(get_last_checkpoint(checkpoint_dir))
        ModelCheckpoint.load_objects(to_load=to_load, checkpoint=checkpoint)
        starting_epoch = trainer.state.epoch
    else:
        raise ValueError('Unknown runmode, shouldn''t reach this')

    attach_trainer_events(trainer=trainer,
                          evaluator=evaluator,
                          train_data_loader=train_data_loader,
                          test_data_loader=test_data_loader,
                          checkpoint_handler=checkpoint_handler,
                          model=model,
                          summary_writer=summary_writer,
                          eval_freq=train_params["eval_freq"],
                          starting_epoch=starting_epoch,
                          optimizer=optimizer,
                          eval_dir=eval_dir,
                          lr_scheduler=lr_scheduler)

    return trainer

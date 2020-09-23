from torch.optim import Adam, SGD
from dnn.optimizer.build_customized_optimizer import NAG
import torch


class OptimizerFactory():

    @staticmethod
    def create_optimizer(optimizer_name, net_params, optimizer_params) -> torch.optim:
        return globals()[optimizer_name](params=net_params, **optimizer_params)

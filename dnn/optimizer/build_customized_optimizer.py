import torch.optim as optim


class NAG(optim.SGD):
    """
    NAG is a type of SGD
    """
    def __init__(self, params, **optimizer_params):
        super(NAG, self).__init__(params=params, lr=optimizer_params['lr'], momentum=optimizer_params['momentum'],
                                  weight_decay=optimizer_params['weight_decay'], nesterov=True)

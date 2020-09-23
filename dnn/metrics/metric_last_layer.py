from ignite.metrics.metric import Metric
import torch


class MetricLastLayer(Metric):
    def __init__(self, output_transform=lambda x: x):
        super(MetricLastLayer, self).__init__(output_transform=output_transform)
        self.last_layer = []

    def reset(self):
        self.last_layer = []

    def update(self, output):
        self.last_layer += [output]

    def compute(self):
        return torch.cat(self.last_layer).cpu().numpy()

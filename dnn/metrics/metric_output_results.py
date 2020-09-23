from ignite.metrics.metric import Metric
import torch
import numpy as np


class MetricOutputResults(Metric):
    def __init__(self, output_transform=lambda x: x):
        super(MetricOutputResults, self).__init__(output_transform=output_transform)
        self.y = []
        self.y_pred = []
        self.img_name = []
        self.confidence = []

    def reset(self):
        self.y = []
        self.y_pred = []
        self.img_name = []
        self.confidence = []

    def update(self, output):
        self.y += [output[1]]
        self.y_pred += [output[0]]
        self.confidence += [output[2]]
        self.img_name += output[3]

    def compute(self):
        return torch.cat(self.y_pred).cpu().numpy(), torch.cat(self.y).cpu().numpy(), \
               torch.cat(self.confidence).cpu().numpy(), self.img_name

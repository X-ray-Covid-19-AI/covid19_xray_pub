from ignite.utils import convert_tensor
import torch
import numpy as np
from dnn.metrics.metric_last_layer import MetricLastLayer
from dnn.metrics.metric_output_results import MetricOutputResults
from ignite.metrics import Precision, Recall, Accuracy, ConfusionMatrix, Fbeta
from ignite.metrics import Metric
#from ignite.exceptions import NotComputableError
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced
#from sklearn.metrics import roc_auc_score

# class AUROC(Metric):
#
#     def __init__(self, output_transform=lambda x: x):
#         self.y_pred = None
#         self.y_true = None
#         super(AUROC, self).__init__(output_transform=output_transform)
#
#     @reinit__is_reduced
#     def reset(self):
#         self.y_pred, self.y_true = [], []
#         super(AUROC, self).reset()
#
#     @reinit__is_reduced
#     def update(self, output):
#         self.y_pred, self.y_true = output
#         self._num_examples = len(self.y_pred)
#
#     @sync_all_reduce("_num_examples", "_num_correct")
#     def compute(self):
#         # if self._num_examples == 0:
#         #     raise NotComputableError('')
#         return roc_auc_score(self.y_true, self.y_pred)

# We use it if we want to forward more data except to the image and label (some metadata)
def prepare_batch_for_tabolar_mode(batch, device, non_blocking):
    x, y, image_name_list, tabolar_data = batch
    return (convert_tensor(x, device=device, non_blocking=non_blocking),
            convert_tensor(y, device=device, non_blocking=non_blocking),
            list(image_name_list),
            convert_tensor(tabolar_data, device=device, non_blocking=non_blocking).float())


def prepare_batch_for_image_mode(batch, device, non_blocking):
    x, y, image_name_list = batch
    return (convert_tensor(x, device=device, non_blocking=non_blocking),
            convert_tensor(y, device=device, non_blocking=non_blocking),
            list(image_name_list))


def choose_prepare_batch(is_tabolar_mode):
    if is_tabolar_mode == 'yes':
        return prepare_batch_for_tabolar_mode
    return prepare_batch_for_image_mode


def _prepare_batch(batch, device=None, non_blocking=False):
    """Prepare batch for training: pass to a device with options.

    """
    x, y = batch
    return (convert_tensor(x, device=device, non_blocking=non_blocking),
            convert_tensor(y, device=device, non_blocking=non_blocking))


# Finally we use the ignite confusion matrix
def prepare_confusion_matrix(x):
    pred_y, y = x[0], x[1]
    pred_labels = pred_y.argmax(dim=1)
    pred_y = torch.zeros_like(pred_y)
    pred_y[np.arange(pred_y.shape[0]), pred_labels] = 1
    return pred_y, y


def build_metrics():
    metrics = {
        'precision': Precision(lambda x: (x[0].argmax(dim=1), x[1])),
        'recall': Recall(lambda x: (x[0].argmax(dim=1), x[1])),
        'accuracy': Accuracy(lambda x: (x[0].argmax(dim=1), x[1])),
        'confusion_matrix': ConfusionMatrix(2, output_transform=prepare_confusion_matrix),
        'metric_output_results': MetricOutputResults(lambda x: (x[0].argmax(dim=1), x[1], x[0], x[2])),
        'metric_last_layer': MetricLastLayer(lambda x: (x[3])),
        #'auroc':AUROC(lambda x: (x[0], x[1])),
    }
    return metrics

from sklearn.metrics import jaccard_score, precision_score
import numpy as np
from keras import backend as K

def flatten_labels(y):
    return np.argmax(y, axis=-1).flatten()

def flat_iou(y_true, y_pred):
    y_true, y_pred = flatten_labels(y_true), flatten_labels(y_pred)
    mean_iou = 0
    for k in set(y_true):
        component1 = y_true == k
        component2 = y_pred == k
        overlap = component1 * component2  # Logical AND
        union = component1 + component2  # Logical OR
        mean_iou += overlap.sum() / float(union.sum())
    mean_iou = mean_iou/(len(set(y_true)))
    return mean_iou

def flat_overall_precision(y_true, y_pred):
    y_true, y_pred = flatten_labels(y_true), flatten_labels(y_pred)
    return precision_score(y_true, y_pred, average='micro')

def flat_per_class_precision(y_true, y_pred):
    y_true, y_pred = flatten_labels(y_true), flatten_labels(y_pred)
    return precision_score(y_true, y_pred, average='macro')

def overall_precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

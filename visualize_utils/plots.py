import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
RUN_NAME = "v3"
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, precision_score, auc, plot_precision_recall_curve

results_dir = "../dnn/configs/log_v3/eval/"

def get_labels_and_probs(dir, filename = "results.csv", true_labels_col = "gt", probs_col = "no_corona_confidence"):
    results = pd.read_csv(os.path.join(dir, filename))

    probs = results[probs_col].values
    true = results[true_labels_col].values

    return true, probs

def plot_roc(dir, filename = "results.csv", true_labels_col = "gt", probs_col = "no_corona_confidence", color = 'dodgerblue'):
    true, probs = get_labels_and_probs(dir, filename, true_labels_col, probs_col)

    fpr, tpr, thresholds = roc_curve(true, probs)
    roc_auc = auc(fpr, tpr)
    print("roc-auc score is", roc_auc)

    plt.figure()
    plt.plot(fpr, tpr, color=color,lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

def plot_precision_recall(dir, filename = "results.csv", true_labels_col = "gt", probs_col = "no_corona_confidence", color = 'dodgerblue'):
    true, probs = get_labels_and_probs(dir, filename, true_labels_col, probs_col)
    precision, recall, thresholds = precision_recall_curve(true, probs)
    pr_auc = auc(recall, precision)
    print("precision-recall auc score is", pr_auc)
    plt.plot(recall, precision, marker='.', label='Logistic')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    plot_roc(results_dir)
    plot_precision_recall(results_dir)


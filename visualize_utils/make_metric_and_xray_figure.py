import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_curve, auc, precision_recall_curve
from matplotlib.font_manager import FontProperties
import os.path as osp

def xray_figure(df, dir_path):
    """
    Function to produce one figure of 3 x-ray images, their confidence score and the ground truth.
    The images are chosen randomly one from each condition of confidence value: above 0.9, below 0.1, between 0.4 and 0.6
    """

    # return one random line from df by condition
    def get_img_by_condition(df, cond):
        return df[cond].sample()

    df['gt_name'] = ['positive for COVID-19' if g == 0 else 'negative for COVID-19' for g in df['gt']]

    # get values
    # option 1: positive, negative and challenging images
    #     high_conf = get_img_by_condition(df, df.corona_confidence > 0.9)
    #     low_conf = get_img_by_condition(df, df.corona_confidence < 0.1)
    #     middle_conf = get_img_by_condition(df, (df.corona_confidence.between(0.4,0.6)))
    #     samples = [high_conf, low_conf, middle_conf]

    # option 2: 3 challenging images
    challenge1 = get_img_by_condition(df, (df.corona_confidence.between(0.4, 0.6)))
    challenge2 = get_img_by_condition(df, (df.corona_confidence.between(0.4, 0.6)))
    challenge3 = get_img_by_condition(df, (df.corona_confidence.between(0.4, 0.6)))
    samples = [challenge1, challenge2, challenge3]

    positions = [1 / 4, 0.51, 0.79]

    # create figure
    fig = plt.figure(figsize=(20, 8))
    columns = 3
    rows = 1

    for i, sample in enumerate(samples):
        img_path = sample.iloc[0]['image_name']
        conf = sample.iloc[0]['corona_confidence']
        gt_name = sample.iloc[0]['gt_name']

        fig.add_subplot(rows, columns, i + 1)
        img = np.asarray(Image.open(img_path))
        plt.imshow(img[:, :, 2], cmap='gray')
        plt.axis('off')
        txt = "Confidence: {:.2f} \nGT: {}".format(conf, gt_name)
        #         txt1 = "Ground Truth: {}".format(gt)
        plt.figtext(positions[i], 0.12, txt, wrap=True,
                    verticalalignment='bottom', horizontalalignment='center',
                    fontsize=20)
    fig.suptitle('3 examples of images and their confidence score', y=0.85, fontsize=18)
    plt.savefig(fname=osp.join(dir_path, 'images_with_conf.png'), format="png", bbox_inches="tight")

# def create_metric_table(df_train, df_test, dir_path):
def create_metric_table(df_test, dir_path):
    """
    Create .png table with accuracy, precision and recall of train and test.
    uses the df 'results'
    """
    # helper function calculate and return accuracy, precision and recall of df
    # need df with 'gt' and 'net_prediction' columns.
    def get_metrics(df):
        gt = list(df['gt'])
        predicted = list(df['net_prediction'])

        accuracy = accuracy_score(gt, predicted).round(2)
        precision = precision_score(gt, predicted).round(2)
        recall = recall_score(gt, predicted).round(2)

        return accuracy, precision, recall

    # train_accuracy, train_precision, train_recall = get_metrics(df_train) # train df
    test_accuracy, test_precision, test_recall = get_metrics(df_test) #test df

    # data for table
    columns = ('Accuracy', 'Precision', 'Recall')
    rows = ['Test']
    # data = [[train_accuracy, train_precision, train_recall],
    #         [test_accuracy, test_precision, test_recall]]
    data = [[test_accuracy, test_precision, test_recall]]
    
    # make table figure
    fig = plt.figure(figsize=(6, 2))
    ytable = plt.table(cellText=data,
                       cellLoc = 'center',
                       rowLabels=rows,
                       colLabels=columns,
                       loc='center')
    ytable.set_fontsize(15)
    ytable.scale(1, 3)
    for (row, col), cell in ytable.get_celld().items():
        if (row == 0) or (col == -1):
            cell.set_text_props(fontproperties=FontProperties(weight='bold'))

    plt.axis("off")
    plt.title('Metrics scores')
    plt.savefig(fname=osp.join(dir_path, 'metric_table.png'), format="png", bbox_inches="tight")

def plot_roc(df, dir_path, true_labels_col = "gt", probs_col = "corona_confidence", color = 'dodgerblue', show = False):
    probs = df[probs_col].values
    true = df[true_labels_col].values

    fpr, tpr, thresholds = roc_curve(true, probs)
    roc_auc = auc(fpr, tpr)
    print("roc-auc score is", roc_auc)

    plt.figure()
    plt.plot(fpr, tpr, color=color,lw=2, label = "roc curve, AUC={:.2f}".format(roc_auc))
    plt.plot([0, 1], [0, 1], color='indigo', linestyle='--', label = 'random guess')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.grid()
    if show:
        plt.show()
    else:
        plt.savefig(fname=osp.join(dir_path, 'roc_curve.png'), format="png", bbox_inches="tight")
    plt.clf()

def plot_precision_recall(df, dir_path, true_labels_col = "gt", probs_col = "corona_confidence", color = 'dodgerblue'):
    probs = df[probs_col].values
    true = df[true_labels_col].values
    true = -1*(true -1) # convert 0 to 1 and vice versa
    probs = 1 - probs #invert probs
    precision, recall, thresholds = precision_recall_curve(true, probs)
    pr_auc = auc(recall, precision)
    print("precision-recall auc score is", pr_auc)
    plt.plot(recall, precision, marker='.', label = 'precision-recall curve, AUC={:.2f}'.format(pr_auc))
    plt.plot([0, 1], [0.5, 0.5], linestyle='--', label='random guess')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid()
    plt.legend(loc="lower left")
    plt.savefig(fname=osp.join(dir_path, 'precision_recall_curve.png'), format="png", bbox_inches="tight")

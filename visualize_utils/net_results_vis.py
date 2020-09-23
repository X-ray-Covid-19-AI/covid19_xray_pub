import base64
import os

import matplotlib.pyplot as plt
from bokeh.io import output_file, save, export_png
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from sklearn.manifold import TSNE
import pandas as pd
import os.path as osp
from bokeh.palettes import Category20

import numpy as np

def plot_tsne(features_df, dir_path, hover=False):
    unique_gt = features_df['gt'].unique()
    unique_gtnames = ['positive for Covid-19' if g == 1 else 'negative for Covid-19' for g in unique_gt]
    colors = Category20[20]

    features_df['gt_name'] = ['positive for Covid-19' if g == 1 else 'negative for Covid-19' for g in features_df['gt']]

    if hover:
        features_df['images_b64'] = [base64.b64encode(open(img, 'rb').read()).decode('utf-8') for img in
                                     features_df['image_path']]
        tips = """
                <div>
                    <div>
                        <img
                            src="data:image/jpeg;base64,@images_b64" height=150 width=150
                            style="float: left; margin: 0px 2px 2px 0px"
                            border="2"
                        ></img>
                    </div>

                    <div>
                        <span style="font-size: 15px;">@gt_name</span>
                    </div>
                    <div>
                        <span style="font-size: 15px;">@score</span>
                    </div>
                </div>
            """
    else:
        tips = """
                <div>
                    <div>
                        <span style="font-size: 50px;">@gt_name</span>
                    </div>
                    <div>
                        <span style="font-size: 50px;">@net_corona_confidence</span>
                    </div>
                    <div>
                        <span style="font-size: 35px;">@image_name</span>
                    </div>
                </div>
            """

    fig_to_plot = figure(width=2000, height=1500, tooltips=tips)

    # Usable if we want to give a sense of confidence
    # features_df['error'] = abs(features_df['gt'] - features_df['net_no_corona_confidence'])
    # features_df['uncertain'] = 2 * (0.5 - abs(0.5 - features_df['net_no_corona_confidence']))
    # features_df['line_alpha_error'] = 0.1 + 0.9 * features_df['error']
    # features_df['line_alpha_uncertain'] = 0.1 + 0.9 * features_df['uncertain']

    for label, label_name in zip(unique_gt, unique_gtnames):
        df_per_label = features_df[features_df['gt'] == label]
        source = ColumnDataSource(df_per_label)

        # Usable if we want to give a sense of confidence
        # fig_to_plot.scatter('tsne0', 'tsne1', color=colors[label * 19], source=source,
        #                     legend_label='legend_label: ' + str(label),
        #                     fill_alpha='uncertain', line_alpha='line_alpha_uncertain', size=30)

        fig_to_plot.scatter('tsne0', 'tsne1', color=colors[label * 19], source=source,
                            legend_label=label_name, size=30)

    fig_to_plot.legend.location = 'bottom_left'
    fig_to_plot.legend.click_policy = 'hide'
    fig_to_plot.legend.label_text_font_size = '50pt'

    html_path = osp.join(dir_path, '{}.html'.format('tnse_plot'))
    if osp.exists(html_path):
        os.remove(html_path)

    output_file(html_path, mode='inline')
    save(fig_to_plot)

    # save tsne as png
    for label, label_name in zip(unique_gt, unique_gtnames):
        df_per_label = features_df[features_df['gt'] == label]
        plt.scatter(df_per_label['tsne0'], df_per_label['tsne1'], c=colors[label * 19],
                    label=label_name, alpha=0.4, s=30)
        plt.legend(scatterpoints=1, loc = "lower left")
        plt.title('T-SNE plot')
        plt.xlabel('tsne0')
        plt.ylabel('tsne1')
    plt.savefig(fname=osp.join(dir_path, 'tsne_plot.png'), format="png", bbox_inches="tight")

def find_knn(feature_map, k=3, algo='ball_tree'):
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=(k+1), algorithm=algo).fit(feature_map)
    distances, indices = nbrs.kneighbors(feature_map)
    indices = indices[:, 1:] #ignore the first index which is the image itself
    distances = distances[:, 1: ] #ignore the first index which is the distance to the image itself
    return indices, distances

def percent_knn_same_label(nn_indices, labels, k =3):
    percents = []
    for ind, point in enumerate(nn_indices):
        cur_label = labels[ind]
        num_same_label = np.sum([labels[jnd] == cur_label for jnd in nn_indices[ind, :]])
        percents.append(num_same_label/float(k))
    return np.array(percents)

def knn_dict(knn_indices, img_names):
    knn_dict = {}
    for ind, nneighbors in enumerate(knn_indices):
        cur_img = img_names[ind]
        knn_dict[cur_img] = [img_names[neighbor_img] for neighbor_img in nneighbors]
    return knn_dict

def knn_predictions(knn_indices, labels, k=3):
    preds = []
    for ind, nneighbors in enumerate(knn_indices):
        cur_neighbors = knn_indices[ind, :]
        num_labeled_1 = np.sum([labels[jnd] == 1 for jnd in cur_neighbors])
        preds.append(num_labeled_1/float(k))
    preds_binary = [int(k > 0.5) for k in preds]
    return preds_binary, preds

def knn_analysis(feature_map, results, dir_path):
    from sklearn.metrics import roc_auc_score, accuracy_score, precision_score
    import pickle

    knn_preds_csv_path = osp.join(dir_path, 'knn_predictions.csv')
    y_pred, y, conf, img_name = results
    df = pd.DataFrame()

    for k_val in range(2,6):
        knn_indices, knn_distances = find_knn(feature_map, k_val)
        df["knn_" + str(k_val) + "_binary_pred"], df["knn_" + str(k_val) + "_pred"] = knn_predictions(knn_indices, y, k=k_val)
        knn_ims = knn_dict(knn_indices, img_name)
        with open(osp.join(dir_path, 'knn_' + str(k_val)+'.pkl'), "wb") as f:
            pickle.dump(knn_ims, f)
    df.to_csv(knn_preds_csv_path)

    knn_analysis_csv_path = osp.join(dir_path, 'knn_analysis.csv')
    analysis_df = pd.DataFrame()
    for k_val in range(2,6):
        analysis_df['knn_' + str(k_val) + '_auroc_k_' ] = [roc_auc_score(y_true=y, y_score=df["knn_" + str(k_val) + "_pred"])]
        analysis_df['knn_' + str(k_val) + '_accuracy_k'] = [accuracy_score(y_true=y, y_pred=df["knn_" + str(k_val) + "_binary_pred"])]
        analysis_df['knn_' + str(k_val) + '_precision_k'] = [precision_score(y_true=y, y_pred=df["knn_" + str(k_val) + "_binary_pred"])]
    analysis_df.to_csv(knn_analysis_csv_path)

def tsne(feature_map, results, component_num, dir_path):
    # TODO: optional - to do PCA before TSNE
    fig, ax = plt.subplots()
    y_pred, y, conf, img_name = results
    model = TSNE(n_components=component_num, random_state=40)
    model.fit(feature_map)
    embeddings = model.embedding_
    df = pd.DataFrame()
    df['tsne0'] = embeddings[:, 0]
    df['tsne1'] = embeddings[:, 1]
    df['gt'] = y
    # df['net_label_prediction'] = results[]
    df['net_corona_confidence'] = conf[:, 0]
    df['net_no_corona_confidence'] = conf[:, 1]
    df['image_name'] = img_name

    df_path = osp.join(dir_path, "tsne_results.csv")
    df.to_csv(df_path)

    for label in [0, 1]:  # TODO: change to df.gt.unique
        df_per_label = df[df['gt'] == label]
        plt.scatter(df_per_label['tsne0'], df_per_label['tsne1'], s=1)
    plot_tsne(features_df=df, dir_path=dir_path)
    return fig

if __name__ == '__main__':
    import pickle
    pth = "C:\\Users\\Alacrity\\Documents\\GitHub\\covid19_xray\\"

    with open(pth + "img_embeddings.pkl", 'rb') as f:
        embeddings_tst = pickle.load(f)

    with open(pth + "img_embeddings_train_no_aug.pkl", 'rb') as f:
        embeddings_trn = pickle.load(f)

    from sklearn.metrics import recall_score, classification_report, roc_auc_score, accuracy_score, precision_score
    from sklearn.neighbors import NearestNeighbors

    trn = embeddings_trn['embeddings']
    tst = embeddings_tst['embeddings']

    trn_labels = embeddings_trn['label']
    tst_labels = embeddings_tst['label']

    for k in range(2,5):
        nbrs = NearestNeighbors(n_neighbors=(k), algorithm='ball_tree').fit(trn)
        distances, indices = nbrs.kneighbors(tst)
        bin_preds, preds = knn_predictions(indices, trn_labels)
        print("k=",k)
        print(classification_report(tst_labels, bin_preds))
        print("auroc =", roc_auc_score(tst_labels, preds))

    print(embeddings_trn.keys())
import os
import os.path as osp
import matplotlib.pyplot as plt


def save_graphs(dir_path, graph, name):
    os.makedirs(dir_path, exist_ok=True)
    graph.savefig(fname=osp.join(dir_path, f'cm_matrix_{name}.png'), format="png", bbox_inches="tight")
    plt.close(graph)

from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import numpy as np


def tsne_visualizer(data_set, representation):

    data_path = './data/corpus.shuffled'

    with open(f'{data_path}/split_index/{data_set}.train', 'r') as f:
        lines = f.readlines()
    train_size = len(lines)

    with open(f'{data_path}/meta/{data_set}.meta', 'r') as f:
        lines = f.readlines()
    target_names = set()
    labels = []
    for line in lines:
        line = line.strip()
        temp = line.split('\t')
        labels.append(temp[2])
        target_names.add(temp[2])

    target_names = list(target_names)

    with open(f'./data/{data_set}_doc_vectors.txt', 'r') as f:
        lines = f.readlines()
    docs = []
    for line in lines:
        temp = line.strip().split()
        values_str_list = temp[1:]
        values = [float(x) for x in values_str_list]
        docs.append(values)

    fea = docs[train_size:]
    label = labels[train_size:]
    label = np.array(label)

    fea = TSNE(n_components=2).fit_transform(fea)
    cls = np.unique(label)

    fea_num = [fea[label == i] for i in cls]
    for i, f in enumerate(fea_num):
        if cls[i] in range(10):
            plt.scatter(f[:, 0], f[:, 1], label=cls[i], marker='+')
        else:
            plt.scatter(f[:, 0], f[:, 1], label=cls[i])

    plt.legend(ncol=5, loc='upper center',
               bbox_to_anchor=(0.48, -0.08), fontsize=11)
    plt.tight_layout()
    plt.savefig(
        f'logs/tsne/{representation}_dataset_{data_set}.png', dpi=150)

    plt.close()

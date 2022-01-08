from sys import argv

from trainer.configs import TrainingConfigs
from trainer.train_model import train_model

from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import numpy as np
#import os
#from common import get_hyperparameters


def create_training_cfg() -> TrainingConfigs:
    # 20NG - MR - Ohsumed - R8, R52
    conf = TrainingConfigs()
    conf.data_sets = ['20ng', 'R8', 'R52', 'ohsumed',
                      'mr', 'cora', 'citeseer', 'pubmed', 'test']
    #conf.data_sets = ['20ng', 'R8', 'R52', 'ohsumed', 'mr', 'test']
    conf.corpus_split_index_dir = 'data/corpus.shuffled/split_index/'
    conf.corpus_node_features_dir = 'data/corpus.shuffled/node_features/'
    conf.corpus_adjacency_dir = ''  # 'data/corpus.shuffled/adjacency/'
    conf.corpus_vocab_dir = 'data/corpus.shuffled/vocabulary/'
    conf.adjacency_sets = ['default', 'syntactic', 'semantic', 'graph']
    conf.model = 'gcn'
    conf.learning_rate = 0.02
    conf.epochs = 200
    conf.hidden1 = 200
    conf.dropout = 0.5
    conf.weight_decay = 0.
    conf.early_stopping = 10
    conf.chebyshev_max_degree = 3
    conf.build()
    return conf


def train(ds: str, training_cfg: TrainingConfigs):
    # Start training
    return train_model(ds_name=ds, is_featureless=True, cfg=training_cfg)


def save_history(hist, representation, dataset):
    file_name = f'experiments/{representation}_dataset_{dataset}.txt'

    with open(file_name, 'w') as my_file:
        my_file.writelines(hist)


def tsne_visualizer(data_set, representation):
    # data_set = 'mr' # 20ng R8 R52 ohsumed mr
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

    fea = docs[train_size:]      # int(train_size * 0.9)
    label = labels[train_size:]  # int(train_size * 0.9)
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
        f'experiments/{representation}_dataset_{data_set}.png', dpi=300)
    #plt.savefig(f'./experiments/{representation}/{data_set}/EXPERIMENT_{experiment}_RUN_{run_time}.png', dpi=300)
    plt.close()


def batch_train(dataset: str, rp: str, trn_cfg):
    '''
    Experiments > Graph Representation > Model Hyperparameter Tuning > Run Step
    '''

    #hyperparameters = get_hyperparameters()
    path = 'data/corpus.shuffled/adjacency/'

    if rp == 'default':
        # Default adjacency
        trn_cfg.corpus_adjacency_dir = f'{path}/default/'
    elif rp == 'syntactic_dependency':
        # Syntactic adjacency
        trn_cfg.corpus_adjacency_dir = f'{path}/syntactic_dependency/'
    elif rp == 'linguistic_inquiry':
        # Semantic adjacency
        trn_cfg.corpus_adjacency_dir = f'{path}/linguistic_inquiry/'
    elif rp == 'graph':
        # Graph adjacency
        trn_cfg.corpus_adjacency_dir = f'{path}/graph/'

    for ds in trn_cfg.data_sets:
        print('\n\n░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ '+ds)

        hist = train(ds=ds, training_cfg=trn_cfg)
        save_history(hist, rp, ds)
        #tsne_visualizer(ds, rp)

# def batch_train(dataset: str, rp: str, trn_cfg):
#     '''
#     Experiments > Graph Representation > Model Hyperparameter Tuning > Run Step
#     '''

#     #hyperparameters = get_hyperparameters()
#     path = 'data/corpus.shuffled/adjacency/'

#     if rp == 'default':
#         # Default adjacency
#         trn_cfg.corpus_adjacency_dir = f'{path}/default/'
#     elif rp == 'syntactic':
#         # Syntactic adjacency
#         trn_cfg.corpus_adjacency_dir = f'{path}/syntactic/'
#     elif rp == 'semantic':
#         # Semantic adjacency
#         trn_cfg.corpus_adjacency_dir = f'{path}/semantic/'
#     elif rp == 'graph':
#         # Graph adjacency
#         trn_cfg.corpus_adjacency_dir = f'{path}/graph/'

#     hist = train(ds=ds_name, training_cfg=trn_cfg)
#     save_history(hist, rp, dataset)
#     tsne_visualizer(dataset, rp)


if __name__ == '__main__':

    trn_cfg = create_training_cfg()
    if len(argv) < 3:
        raise Exception(
            "Dataset name cannot be left blank. Must be one of datasets:%r." % trn_cfg.data_sets)

    ds_name = argv[1]
    rp_name = argv[2]

    #print("------ Working with dataset", ds_name, "------\n")
    # ORIGINAL_PAPER = {
    #    "mr": {"avg": 0.7674, "std": 0.0020},
    #    "Ohsumed": {"avg": 0.6836, "std": 0.0056},
    #    "R8": {"avg": 0.9707, "std": 0.0010},
    #    "R52": {"avg": 0.9356, "std": 0.0018}
    # }
    # print(ORIGINAL_PAPER[ds_name])

    batch_train(ds_name, rp_name, trn_cfg)

    print('\nDone!!!')

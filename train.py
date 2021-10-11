from sys import argv

from trainer.configs import TrainingConfigs
from trainer.train_model import train_model

#from matplotlib.backends.backend_pdf import PdfPages
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import numpy as np
import os


def create_training_cfg() -> TrainingConfigs:
    # 20NG - MR - Ohsumed - R8, R52
    conf = TrainingConfigs()
    #conf.data_sets = ['20ng', 'R8', 'R52', 'ohsumed', 'mr', 'cora', 'citeseer', 'pubmed']
    conf.data_sets = ['20ng', 'R8', 'R52', 'ohsumed', 'mr', 'test']
    conf.corpus_split_index_dir = 'data/corpus.shuffled/split_index/'
    conf.corpus_node_features_dir = 'data/corpus.shuffled/node_features/'
    conf.corpus_adjacency_dir = '' #'data/corpus.shuffled/adjacency/'
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


def save_history(hist, representation, dataset, experiment, run_time):
    file_name = f'./experiments/{representation}/{dataset}/EXPERIMENT_{experiment}_RUN_{run_time}.txt'
    if not os.path.exists(f'./experiments/{representation}/{dataset}'):
        os.mkdir(f'./experiments/{representation}/{dataset}')
    my_file = open(file_name, 'w')
    #my_file=map(lambda x:x+'\n', my_file)
    my_file.writelines(hist)
    my_file.close()


def tsne_visualizer(data_set, experiment, run_time, representation):
    # data_set = 'mr' # 20ng R8 R52 ohsumed mr
    data_path = './data/corpus.shuffled'

    #f = open(os.path.join(data_path, data_set + '.train.index'), 'r')
    f = open(f'{data_path}/split_index/{data_set}.train', 'r')
    lines = f.readlines()
    f.close()
    train_size = len(lines)

    #f = open(os.path.join(data_path, data_set + '_shuffle.txt'), 'r')
    f = open(f'{data_path}/meta/{data_set}.meta', 'r')
    lines = f.readlines()
    f.close()

    target_names = set()
    labels = []
    for line in lines:
        line = line.strip()
        temp = line.split('\t')
        labels.append(temp[2])
        target_names.add(temp[2])

    target_names = list(target_names)

    #f = open(os.path.join(data_path, data_set + '_doc_vectors.txt'), 'r')
    f = open(f'./data/{data_set}_doc_vectors.txt', 'r')
    lines = f.readlines()
    f.close()

    docs = []
    for line in lines:
        temp = line.strip().split()
        values_str_list = temp[1:]
        values = [float(x) for x in values_str_list]
        docs.append(values)

    fea = docs[train_size:]    # int(train_size * 0.9)
    label = labels[train_size:]  # int(train_size * 0.9)
    label = np.array(label)

    fea = TSNE(n_components=2).fit_transform(fea)
    #pdf = PdfPages(
    #    f'./experiments/{data_set}_EXPERIMENT_{experiment}_RUN_{run_time}.pdf')
    cls = np.unique(label)

    # cls=range(10)
    fea_num = [fea[label == i] for i in cls]
    for i, f in enumerate(fea_num):
        if cls[i] in range(10):
            plt.scatter(f[:, 0], f[:, 1], label=cls[i], marker='+')
        else:
            plt.scatter(f[:, 0], f[:, 1], label=cls[i])

    plt.legend(ncol=5, loc='upper center', bbox_to_anchor=(0.48, -0.08), fontsize=11)
    # plt.ylim([-20,35])
    # plt.title(md_file)
    plt.tight_layout()
    #pdf.savefig()
    #plt.show()
    #pdf.close()
    plt.savefig(f'./experiments/{representation}/{data_set}/EXPERIMENT_{experiment}_RUN_{run_time}.png', dpi=300)
    plt.close()


def batch_train(dataset: str, rp: str, trn_cfg):
    experiment = 0
    times = 2

    if rp == 'default':
        trn_cfg.corpus_adjacency_dir = 'data/corpus.shuffled/adjacency/default/'  # Default adjacency
    elif rp == 'syntactic':
        trn_cfg.corpus_adjacency_dir = 'data/corpus.shuffled/adjacency/syntactic/'  # Syntactic adjacency
    elif rp == 'semantic':
        pass # semantic adjacency
    else:
        pass

    for indx in range(0, times):
        hist = train(ds=ds_name, training_cfg=trn_cfg)
        save_history(hist, rp, dataset, experiment, indx)
        tsne_visualizer(dataset, experiment, indx, rp)


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

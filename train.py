from trainer.configs import TrainingConfigs
from trainer.train_model import train_model
from tsne import tsne_visualizer
from sys import argv

import matplotlib.pyplot as plt


def create_training_cfg() -> TrainingConfigs:

    conf = TrainingConfigs()
    conf.data_sets = ['20ng', 'R8', 'R52', 'ohsumed',
                      'mr', 'cora', 'citeseer', 'pubmed']
    #conf.data_sets = ['R8']
    conf.corpus_split_index_dir = 'data/corpus.shuffled/split_index/'
    conf.corpus_node_features_dir = 'data/corpus.shuffled/node_features/'
    conf.corpus_adjacency_dir = ''
    conf.corpus_vocab_dir = 'data/corpus.shuffled/vocabulary/'
    conf.adjacency_sets = ['frequency', 'syntactic_dependency',
                           'linguistic_inquiry', 'semantic', 'graph']
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


def create_training_plot(training_history, name="training_history"):
    fig, axes = plt.subplots(2, 1)
    axes[0].plot(training_history.epoch, training_history.accuracy, c="blue")
    axes[0].set_ylabel("Accuracy", size=20)
    axes[0].grid(which="both")
    axes[1].plot(training_history.epoch, training_history.val_loss,
                 c="green", label='Validation')
    axes[1].plot(training_history.epoch,
                 training_history.train_loss, c="red", label='Train')
    axes[1].set_ylabel("Loss", size=20)
    axes[1].set_xlabel("Epoch", size=20)
    axes[1].grid(which="both")
    axes[1].legend(fontsize=15)

    fig = plt.gcf()
    fig.set_size_inches(15, 8)
    plt.tight_layout()
    plt.savefig(f"{name}.jpg", dpi=200)


def batch_train(rp: str, trn_cfg):
    '''
    Experiments > Graph Representation > Model Hyperparameter Tuning > Run Step
    '''

    path = 'data/corpus.shuffled/adjacency/'

    if rp == 'frequency':
        # Default adjacency
        trn_cfg.corpus_adjacency_dir = f'{path}/frequency/'
    elif rp == 'semantic':
        # Semantic adjacency
        trn_cfg.corpus_adjacency_dir = f'{path}/semantic/'
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
        print('\n\n▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄ ' + ds)

        hist = train(ds=ds, training_cfg=trn_cfg)
        save_history(hist, rp, ds)
        tsne_visualizer(ds, rp)


if __name__ == '__main__':

    trn_cfg = create_training_cfg()
    if len(argv) < 2:
        raise Exception(
            "Adjacency Representation name cannot be left blank. Must be one of representation:%r." % trn_cfg.adjacency_sets)

    rp_name = argv[1]

    #print("------ Working with dataset", ds_name, "------\n")
    # ORIGINAL_PAPER = {
    #    "mr": {"avg": 0.7674, "std": 0.0020},
    #    "Ohsumed": {"avg": 0.6836, "std": 0.0056},
    #    "R8": {"avg": 0.9707, "std": 0.0010},
    #    "R52": {"avg": 0.9356, "std": 0.0018}
    # }
    # print(ORIGINAL_PAPER[ds_name])

    batch_train(rp_name, trn_cfg)

    print('\nDone!!!')

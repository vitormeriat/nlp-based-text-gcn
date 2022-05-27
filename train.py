from modules.trainer.configs import TrainingConfigs
from modules.trainer.train_model import train_model
from tsne import tsne_visualizer
import matplotlib.pyplot as plt
from sys import argv
from utils.file_ops import create_dir


def create_training_cfg() -> TrainingConfigs:

    conf = TrainingConfigs()
    #conf.data_sets = ['R8', 'R52', '20ng', 'ohsumed', 'mr', 'cora', 'citeseer', 'pubmed']
    conf.data_sets = ['R8', 'R52', '20ng', 'mr', 'cora', 'citeseer', 'pubmed']
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


def train(ds: str, training_cfg: TrainingConfigs, rp:str):
    # Start training
    return train_model(ds_name=ds, is_featureless=True, cfg=training_cfg, rp=rp)


def save_history(hist, file_name):  # representation, dataset, run_time):
    #file_name = f'logs/experiments/{representation}_dataset_{dataset}.txt'
    file_name = f'{file_name}.txt'

    with open(file_name, 'w') as my_file:
        my_file.writelines(hist)


# representation, dataset, run_time):
def plot_acc_loss(train_acc, eval_acc, train_loss, eval_loss, file_name):
    epochs = range(1, len(train_acc) + 1)

    #file_name = f'logs/experiments/{representation}/RUN_{run_time}/{dataset}'

    plt.figure(figsize=(16, 10))
    plt.plot(epochs, train_acc, 'b', label='Training acc')
    plt.plot(epochs, eval_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.savefig(f'{file_name}-metrics-acc.png', dpi=250)

    plt.figure()
    plt.figure(figsize=(16, 10))
    plt.plot(epochs, train_loss, 'b', label='Training loss')
    plt.plot(epochs, eval_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig(f'{file_name}-metrics-loss.png', dpi=250)


def batch_train(rp: str, trn_cfg):
    '''
    Experiments > Graph Representation > Model Hyperparameter Tuning > Run Step
    '''

    path = 'data/corpus.shuffled/adjacency/'

    for ds in trn_cfg.data_sets:

        print('\n\n'+'▄'*60 + ds + '\n')

        for indx in range(5):
            print('\n\n'+'▄'*30 + f' {indx}\n')

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

            # for ds in trn_cfg.data_sets:
            #     print('\n\n'+'▄'*60 + ds)

            file_name = f'logs/experiments/{rp}/RUN_{indx}'
            create_dir(dir_path=file_name, overwrite=False)
            file_name = f'{file_name}/{ds}'

            hist, train_loss, eval_loss, train_acc, eval_acc = train(
                ds=ds, training_cfg=trn_cfg, rp=rp)
            save_history(hist, file_name)  # rp, ds, indx)
            #tsne_visualizer(ds, rp, indx)
            #create_training_plot(hist, rp, ds, indx)
            plot_acc_loss(train_acc, eval_acc, train_loss,
                          eval_loss, file_name)  # rp, ds, indx)


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

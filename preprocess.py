from preprocessors.build_linguistic_inquiry_adj import build_linguistic_inquiry_adjacency
from preprocessors.build_dependency_adj import build_dependency_adjacency
from preprocessors.build_graph_adjacency import build_graph_adjacency
from preprocessors.build_freq_adjacency import build_freq_adjacency
from preprocessors.build_node_features import build_node_features
from preprocessors.clean_data import clean_data, config_nltk
from preprocessors.configs import PreProcessingConfigs
from preprocessors.prepare_words import prepare_words
from preprocessors.shuffle_data import shuffle_data
from utils.file_ops import create_dir
from utils.logger import PrintLog
from sys import argv


def create_preprocessing_cfg() -> PreProcessingConfigs:
    conf = PreProcessingConfigs()
    conf.data_sets = ['20ng', 'R8', 'R52', 'ohsumed',
                      'mr', 'cora', 'citeseer', 'pubmed', 'test']
    conf.adjacency_sets = ['default', 'syntactic_dependency',
                           'linguistic_inquiry', 'semantic', 'graph']
    conf.data_set_extension = '.txt'
    conf.corpus_dir = 'data/corpus/'
    conf.corpus_meta_dir = 'data/corpus/meta/'
    conf.corpus_cleaned_dir = 'data/corpus.cleaned/'
    conf.corpus_shuffled_dir = 'data/corpus.shuffled/'
    conf.corpus_shuffled_split_index_dir = 'data/corpus.shuffled/split_index/'
    conf.corpus_shuffled_meta_dir = 'data/corpus.shuffled/meta/'
    conf.corpus_shuffled_vocab_dir = 'data/corpus.shuffled/vocabulary/'
    conf.corpus_shuffled_word_vectors_dir = 'data/corpus.shuffled/word_vectors/'
    conf.corpus_shuffled_adjacency_dir = 'data/corpus.shuffled/adjacency/'
    conf.corpus_shuffled_node_features_dir = 'data/corpus.shuffled/node_features/'
    conf.core_nlp_path = 'C:/bin/CoreNLP/stanford-corenlp-full-2018-10-05'
    conf.liwc_path = 'C:/bin/LIWC/LIWC2007_English100131.dic'
    conf.build()
    return conf


def save_history(hist, representation, dataset):
    create_dir(dir_path='data/log', overwrite=False)
    with open(f'data/log/{representation}_dataset_{dataset}.txt', 'w') as my_file:
        my_file.writelines(hist)


def preprocess(ds: str, rp: str, preprocessing_cfg: PreProcessingConfigs):  # Start pre-processing
    pl = PrintLog()
    config_nltk()
    clean_data(ds_name=ds, rare_count=5, cfg=preprocessing_cfg, pl=pl)
    shuffle_data(ds_name=ds, cfg=preprocessing_cfg, pl=pl)
    prepare_words(ds_name=ds, cfg=preprocessing_cfg, pl=pl)
    build_node_features(ds_name=ds, validation_ratio=0.10,
                        use_predefined_word_vectors=False, cfg=preprocessing_cfg, pl=pl)

    if rp == 'frequency':
        build_freq_adjacency(
            ds_name=ds, cfg=preprocessing_cfg, pl=pl)  # Frequency adjacency
    elif rp == 'syntactic_dependency':
        build_dependency_adjacency(
            ds_name=ds, cfg=preprocessing_cfg, pl=pl)  # Dependency adjacency
    elif rp == 'linguistic_inquiry':
        build_linguistic_inquiry_adjacency(
            ds_name=ds, cfg=preprocessing_cfg, pl=pl)  # Linguistic Inquiry and Word Count
    elif rp == 'graph':
        build_graph_adjacency(
            ds_name=ds, cfg=preprocessing_cfg, pl=pl)  # Graph adjacency

    hist = pl.log_history()
    save_history(hist, rp, ds)


def batch_preprocessing(rp: str, preprocessing_cfg: PreProcessingConfigs):
    for ds in preprocessing_cfg.data_sets:
        print('\n\n▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄ ' + ds)
        preprocess(ds, rp, preprocessing_cfg)


if __name__ == '__main__':
    prep_cfg = create_preprocessing_cfg()
    if len(argv) < 2:
        raise Exception(
            "Dataset name cannot be left blank. Must be one of datasets:%r." % prep_cfg.data_sets)

    rp_name = argv[1]
    #preprocess(ds=ds_name, rp=rp_name, preprocessing_cfg=prep_cfg)
    batch_preprocessing(rp_name, prep_cfg)

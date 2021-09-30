import string
from stanfordcorenlp import StanfordCoreNLP
from nltk.corpus import stopwords
import numpy as np
from math import log
import scipy.sparse as sp
import pickle as pkl
from preprocessors.configs import PreProcessingConfigs
from common import check_data_set
from utils.file_ops import create_dir, check_paths





def build_syntactic_adjacency(ds_name: str, cfg: PreProcessingConfigs):

    # input files
    ds_corpus = cfg.corpus_shuffled_dir + ds_name + ".txt"
    ds_corpus_vocabulary = cfg.corpus_shuffled_vocab_dir + ds_name + '.vocab'
    ds_corpus_train_idx = cfg.corpus_shuffled_split_index_dir + ds_name + '.train'
    ds_corpus_test_idx = cfg.corpus_shuffled_split_index_dir + ds_name + '.test'

    # checkers
    check_data_set(data_set_name=ds_name, all_data_set_names=cfg.data_sets)
    check_paths(ds_corpus, ds_corpus_vocabulary, ds_corpus_train_idx, ds_corpus_test_idx)

    create_dir(dir_path=cfg.corpus_shuffled_adjacency_dir, overwrite=False)

    docs_of_words = [line.split() for line in open(file=ds_corpus)]
    vocab = open(ds_corpus_vocabulary).read().splitlines()  # Extract Vocabulary.
    word_to_id = {word: i for i, word in enumerate(vocab)}  # Word to its id.
    train_size = len(open(ds_corpus_train_idx).readlines())  # Real train-size, not adjusted.
    test_size = len(open(ds_corpus_test_idx).readlines())  # Real test-size.

    core_nlp_path = cfg.corpus_shuffled_dir
    nlp = StanfordCoreNLP(core_nlp_path, lang='en')
    stop_words = set(stopwords.words('english'))
    pass
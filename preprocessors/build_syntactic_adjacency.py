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
from typing import List, Dict, Tuple



def extract_windows(docs_of_words: List[List[str]], window_size: int) -> List[List[str]]:
    """Word co-occurrence with context windows"""
    windows = []
    for doc_words in docs_of_words:
        doc_len = len(doc_words)
        if doc_len <= window_size:
            windows.append(doc_words)
        else:
            for j in range(doc_len - window_size + 1):
                window = doc_words[j: j + window_size]
                windows.append(window)
    return windows



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

    window_size=20

    windows_of_words = extract_windows(docs_of_words=docs_of_words, window_size=window_size)

    
    
    rela_pair_count_str = {}
    for doc_id in range(len(docs_of_words)):
        print(doc_id)
        words = docs_of_words[doc_id]
        words = words.split("\n")
        rela=[]
        for window in words:
            if window==' ':
                continue
            # Construir rela_pair_count
            window = window.replace(string.punctuation, ' ')
            res = nlp.dependency_parse(window)
            for tuple in res:
                rela.append(f'{tuple[0]}, {tuple[1]}')
            for pair in rela:
                pair=pair.split(", ")
                if pair[0]=='ROOT' or pair[1]=='ROOT':
                    continue
                if pair[0] == pair[1]:
                    continue
                if pair[0] in string.punctuation or pair[1] in string.punctuation:
                    continue
                if pair[0] in stop_words or pair[1] in stop_words:
                    continue
                word_pair_str = pair[0] + ',' + pair[1]
                if word_pair_str in rela_pair_count_str:
                    rela_pair_count_str[word_pair_str] += 1
                else:
                    rela_pair_count_str[word_pair_str] = 1
                # two orders
                word_pair_str = pair[1] + ',' + pair[0]
                if word_pair_str in rela_pair_count_str:
                    rela_pair_count_str[word_pair_str] += 1
                else:
                    rela_pair_count_str[word_pair_str] = 1

    
    #ds_syntactic = cfg.corpus_shuffled_dir + '/{}_stan.pkl'.format(ds_name) + ".txt"
    #output0=open(ds_syntactic,'wb')
    #pkl.dump(rela_pair_count_str, output0)
    #output0.close()

    data1 = rela_pair_count_str
    del(rela_pair_count_str)

    max_count1 = 0.0
    min_count1 = 0.0
    count1 = []
    for key in data1:
        if data1[key] > max_count1:
            max_count1 = data1[key]
        if data1[key] < min_count1:
            min_count1 = data1[key]
        count1.append(data1[key])
    count_mean1 = np.mean(count1)
    count_var1 = np.var(count1)
    count_std1 = np.std(count1, ddof=1)


    # compute weights
    for key in word_pair_count:
        temp = key.split(',')
        i = int(temp[0])
        j = int(temp[1])
        count = word_pair_count[key]
        word_freq_i = word_window_freq[vocab[i]]
        word_freq_j = word_window_freq[vocab[j]]
        pmi = log((1.0 * count / window_size) / (1.0 * word_freq_i * word_freq_j / (window_size * window_size)))
        if pmi <= 0:
            continue
        # pmi
        row.append(train_size + i)
        col.append(train_size + j)
        ## weight.append(pmi)
        # Dependência sintática
        if i not in id_word_map or j not in id_word_map:
            continue
        newkey = id_word_map[i] + ',' + id_word_map[j]
        if newkey in data1:
            # padronização min-max
            wei = (data1[newkey] - min_count1) / (max_count1 - min_count1)
            # 0 normalização média
            # wei = (data1[key]-count_mean1)/ count_std1
            # Taxa de frequência de ocorrência, mais frequentemente quando 1 aparece
            # wei = data1[key] / data2[key]
            weight1.append(wei)
        else:
            weight1.append(pmi)
    
    pass
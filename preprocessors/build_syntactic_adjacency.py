import string
from time import time
from stanfordcorenlp import StanfordCoreNLP
from nltk.corpus import stopwords
import numpy as np
from math import log
import scipy.sparse as sp
import pickle as pkl
from preprocessors.configs import PreProcessingConfigs
from common import check_data_set
from utils.file_ops import create_dir, check_paths

from preprocessors.adjacency import extract_windows

def remove_punctuation(text):
    for punc in list(string.punctuation):
        if punc in text:
            text = text.replace(punc, ' ')
    return text.strip()

def build_syntactic_adjacency(ds_name: str, cfg: PreProcessingConfigs):

    t1 = time()
    # input files
    ds_corpus = cfg.corpus_shuffled_dir + ds_name + ".txt"
    ds_corpus_vocabulary = cfg.corpus_shuffled_vocab_dir + ds_name + '.vocab'
    ds_corpus_train_idx = cfg.corpus_shuffled_split_index_dir + ds_name + '.train'
    ds_corpus_test_idx = cfg.corpus_shuffled_split_index_dir + ds_name + '.test'

    # checkers
    check_data_set(data_set_name=ds_name, all_data_set_names=cfg.data_sets)
    check_paths(ds_corpus, ds_corpus_vocabulary,
                ds_corpus_train_idx, ds_corpus_test_idx)

    create_dir(dir_path=cfg.corpus_shuffled_adjacency_dir + "/syntactic", overwrite=False)

    docs_of_words = [line.split() for line in open(file=ds_corpus)]
    # Extract Vocabulary.
    vocab = open(ds_corpus_vocabulary).read().splitlines()
    word_to_id = {word: i for i, word in enumerate(vocab)}  # Word to its id.
    # Real train-size, not adjusted.
    train_size = len(open(ds_corpus_train_idx).readlines())
    test_size = len(open(ds_corpus_test_idx).readlines())  # Real test-size.

    #core_nlp_path = cfg.corpus_shuffled_dir
    nlp = StanfordCoreNLP(cfg.core_nlp_path, lang='en')
    stop_words = set(stopwords.words('english'))

    window_size = 20

    #windows_of_words = extract_windows(docs_of_words=docs_of_words, window_size=window_size)

    # windows = []
    # for doc_words in docs_of_words:
    #     #words = doc_words.split()
    #     length = len(doc_words)
    #     if length <= window_size:
    #         windows.append(doc_words)
    #     else:
    #         # print(length, length - window_size + 1)
    #         for j in range(length - window_size + 1):
    #             window = doc_words[j: j + window_size]
    #             windows.append(window)
    #             # print(window)

    # windows_of_words=windows
    # del(windows)

    # =============================================================
    word_doc_list = {}

    for i in range(len(docs_of_words)):
        #doc_words = docs_of_words[i]
        #words = doc_words.split()
        words = docs_of_words[i]
        appeared = set()
        for word in words:
            if word in appeared:
                continue
            if word in word_doc_list:
                doc_list = word_doc_list[word]
                doc_list.append(i)
                word_doc_list[word] = doc_list
            else:
                word_doc_list[word] = [i]
            appeared.add(word)
    # =============================================================
    word_doc_freq = {}
    for word, doc_list in word_doc_list.items():
        word_doc_freq[word] = len(doc_list)
    # =============================================================
    vocab_size = len(vocab)
    word_id_map = {}
    id_word_map = {}
    for i in range(vocab_size):
        word_id_map[vocab[i]] = i
        id_word_map[i] = vocab[i]
    # =============================================================
    word_window_freq = {}
    for window in docs_of_words:
        appeared = set()
        for i in range(len(window)):
            if window[i] in appeared:
                continue
            if window[i] in word_window_freq:
                word_window_freq[window[i]] += 1
            else:
                word_window_freq[window[i]] = 1
            appeared.add(window[i])
    # =============================================================
    word_pair_count = {}
    for window in docs_of_words:
        for i in range(1, len(window)):
            for j in range(0, i):
                word_i = window[i]
                word_i_id = word_id_map[word_i]
                word_j = window[j]
                word_j_id = word_id_map[word_j]
                if word_i_id == word_j_id:
                    continue
                word_pair_str = str(word_i_id) + ',' + str(word_j_id)
                if word_pair_str in word_pair_count:
                    word_pair_count[word_pair_str] += 1
                else:
                    word_pair_count[word_pair_str] = 1
                # two orders
                word_pair_str = str(word_j_id) + ',' + str(word_i_id)
                if word_pair_str in word_pair_count:
                    word_pair_count[word_pair_str] += 1
                else:
                    word_pair_count[word_pair_str] = 1
    # =============================================================
    rela_pair_count_str = {}
    for doc_id in range(len(docs_of_words)):
        #print(doc_id)
        words = docs_of_words[doc_id]
        sentence = ' '.join(words)
        
        #words = words.split("\n")
        #rela=[]
        # for window in words:
        #     if window==' ':
        #         continue
        # Construir rela_pair_count
        #window = window.replace(string.punctuation, ' ')
        res = nlp.dependency_parse(sentence)
        tokenized = nlp.word_tokenize(sentence)
        #print(res)
        #print(tokenized)
        #for tuple in res:
        #    #rela.append(f'{tuple[0]}, {tuple[1]}')
        #    rela.append(f'{tuple[0]}, {tuple[1]}, {tuple[2]}')
        for pair in list(res):
            #pair=pair.split(", ")
            if pair[0]=='ROOT' or pair[1]=='ROOT':
                continue
            if pair[0] == pair[1]:
                continue
            #if pair[0] in string.punctuation or pair[1] in string.punctuation:
            #    continue
            if pair[0] in stop_words or pair[1] in stop_words:
                continue
            #word_pair_str = pair[0] + ',' + pair[1]
            #word_pair_str = pair[2] + ',' + pair[1]
            word_pair_str = tokenized[pair[2]-1] + ',' + tokenized[pair[1]-1]
            if word_pair_str in rela_pair_count_str:
                rela_pair_count_str[word_pair_str] += 1
            else:
                rela_pair_count_str[word_pair_str] = 1
            # two orders
            #word_pair_str = pair[1] + ',' + pair[0]
            #word_pair_str = pair[1] + ',' + pair[2]
            word_pair_str = tokenized[pair[1]-1] + ',' + tokenized[pair[2]-1]
            if word_pair_str in rela_pair_count_str:
                rela_pair_count_str[word_pair_str] += 1
            else:
                rela_pair_count_str[word_pair_str] = 1
    # =============================================================
    max_count1 = 0.0
    min_count1 = 0.0
    count1 = []
    for key in rela_pair_count_str:
        if rela_pair_count_str[key] > max_count1:
            max_count1 = rela_pair_count_str[key]
        if rela_pair_count_str[key] < min_count1:
            min_count1 = rela_pair_count_str[key]
        count1.append(rela_pair_count_str[key])
    
    count_mean1 = np.mean(count1)
    count_var1 = np.var(count1)
    count_std1 = np.std(count1, ddof=1)
    # =============================================================
    row = []
    col = []
    weight = []

    # compute weights PMI
    num_window = len(docs_of_words)
    for key in rela_pair_count_str:
        temp = key.split(',')
        i = temp[0]
        j = temp[1]
        count = rela_pair_count_str[key]
        word_freq_i = word_window_freq[i]
        word_freq_j = word_window_freq[j]
        pmi = log((1.0 * count / num_window) / (1.0 * word_freq_i * word_freq_j / (num_window * num_window)))
        #if pmi <= 0:
        #    continue
        # pmi
        row.append(train_size + word_id_map[i])
        col.append(train_size + word_id_map[j])
        #weight.append(pmi)
        if key in rela_pair_count_str:
            #print(':)')
            wei = (rela_pair_count_str[key] - min_count1) / (max_count1 - min_count1)
            # 0 normalização média
            # wei = (rela_pair_count_str[key]-count_mean1)/ count_std1
            weight.append(wei)
        else:
            weight.append(pmi)
    # =============================================================
    # doc word frequency
    weight_tfidf = []
    doc_word_freq = {}
    for doc_id in range(len(docs_of_words)):
        #doc_words = docs_of_words[doc_id]
        #words = doc_words.split()
        words = docs_of_words[doc_id]
        for word in words:
            word_id = word_id_map[word]
            doc_word_str = str(doc_id) + ',' + str(word_id)
            if doc_word_str in doc_word_freq:
                doc_word_freq[doc_word_str] += 1
            else:
                doc_word_freq[doc_word_str] = 1

    for i in range(len(docs_of_words)):
        #doc_words = docs_of_words[i]
        #words = doc_words.split()
        words = docs_of_words[i]
        doc_word_set = set()
        for word in words:
            if word in doc_word_set:
                continue
            j = word_id_map[word]
            key = str(i) + ',' + str(j)
            freq = doc_word_freq[key]
            if i < train_size:
                row.append(i)
            else:
                row.append(i + vocab_size)
            col.append(train_size + j)
            idf = log(1.0 * len(docs_of_words) / word_doc_freq[vocab[j]])
            weight_tfidf.append(freq * idf)
            doc_word_set.add(word)
    # =============================================================
    weight = weight + weight_tfidf
    node_size = train_size + vocab_size + test_size
    adj = sp.csr_matrix((weight, (row, col)), shape=(node_size, node_size))
    # =============================================================
    # Dump Adjacency Matrix
    with open(cfg.corpus_shuffled_adjacency_dir + "/syntactic/ind.{}.adj".format(ds_name), 'wb') as f:
        pkl.dump(adj, f)
    # =============================================================
    nlp.close()

    # =============================================================
    elapsed = time() - t1
    print("[INFO] Adjacency Dir='{}'".format(
        cfg.corpus_shuffled_adjacency_dir))
    print("[INFO] Elapsed time is %f seconds." % elapsed)
    print("[INFO] ========= EXTRACTED ADJACENCY MATRIX: Heterogenous doc-word adjacency matrix. =========")










    # # ====================================================
    # word_window_freq = {}
    # for window in windows_of_words:
    #     appeared = set()
    #     for i in range(len(window)):
    #         if window[i] in appeared:
    #             continue
    #         if window[i] in word_window_freq:
    #             word_window_freq[window[i]] += 1
    #         else:
    #             word_window_freq[window[i]] = 1
    #         appeared.add(window[i])


    # # ====================================================
    # rela_pair_count_str = {}
    # for doc_id in range(len(docs_of_words)):
    #     # #print(doc_id)
    #     # words = docs_of_words[doc_id]
    #     # #words = words.split("\n")
    #     # rela = []
    #     # for window in words:
    #     #     if window == ' ':
    #     #         continue

    #     rela = [] 
    #     words = docs_of_words[doc_id]
    #     sentence = ' '.join(words)
    #     # Construir rela_pair_count
    #     #window = window.replace(string.punctuation, ' ')

    #     res = nlp.dependency_parse(sentence)
    #     for tuple in res:
    #         rela.append(f'{tuple[0]}, {tuple[1]}')
    #     for pair in rela:
    #         pair = pair.split(", ")
    #         if pair[0] == 'ROOT' or pair[1] == 'ROOT':
    #             continue
    #         if pair[0] == pair[1]:
    #             continue
    #         if pair[0] in string.punctuation or pair[1] in string.punctuation:
    #             continue
    #         if pair[0] in stop_words or pair[1] in stop_words:
    #             continue
    #         word_pair_str = pair[0] + ',' + pair[1]
    #         if word_pair_str in rela_pair_count_str:
    #             rela_pair_count_str[word_pair_str] += 1
    #         else:
    #             rela_pair_count_str[word_pair_str] = 1
    #         # two orders
    #         word_pair_str = pair[1] + ',' + pair[0]
    #         if word_pair_str in rela_pair_count_str:
    #             rela_pair_count_str[word_pair_str] += 1
    #         else:
    #             rela_pair_count_str[word_pair_str] = 1

    # #ds_syntactic = cfg.corpus_shuffled_dir + '/{}_stan.pkl'.format(ds_name) + ".txt"
    # # output0=open(ds_syntactic,'wb')
    # #pkl.dump(rela_pair_count_str, output0)
    # # output0.close()

    # # ====================================================
    # word_pair_count = {}
    # for window in windows_of_words:
    #     for i in range(1, len(window)):
    #         for j in range(0, i):
    #             word_i = window[i]
    #             word_i_id = word_to_id[word_i]
    #             word_j = window[j]
    #             word_j_id = word_to_id[word_j]
    #             if word_i_id == word_j_id:
    #                 continue
    #             word_pair_str = str(word_i_id) + ',' + str(word_j_id)
    #             if word_pair_str in word_pair_count:
    #                 word_pair_count[word_pair_str] += 1
    #             else:
    #                 word_pair_count[word_pair_str] = 1
    #             # two orders
    #             word_pair_str = str(word_j_id) + ',' + str(word_i_id)
    #             if word_pair_str in word_pair_count:
    #                 word_pair_count[word_pair_str] += 1
    #             else:
    #                 word_pair_count[word_pair_str] = 1

    # # ====================================================
    # #data1 = rela_pair_count_str
    # #del(rela_pair_count_str)

    # max_count1 = 0.0
    # min_count1 = 0.0
    # count1 = []
    # for key in rela_pair_count_str:
    #     if rela_pair_count_str[key] > max_count1:
    #         max_count1 = rela_pair_count_str[key]
    #     if rela_pair_count_str[key] < min_count1:
    #         min_count1 = rela_pair_count_str[key]
    #     count1.append(rela_pair_count_str[key])
    # count_mean1 = np.mean(count1)
    # count_var1 = np.var(count1)
    # count_std1 = np.std(count1, ddof=1)

    # # ====================================================
    # row = []
    # col = []
    # weight1 = []

    # vocab_size = len(vocab)

    # word_doc_list = {}
    # for i in range(len(docs_of_words)):
    #     #doc_words = docs_of_words[i]
    #     #words = doc_words.split()
    #     words = docs_of_words[i]
    #     appeared = set()
    #     for word in words:
    #         if word in appeared:
    #             continue
    #         if word in word_doc_list:
    #             doc_list = word_doc_list[word]
    #             doc_list.append(i)
    #             word_doc_list[word] = doc_list
    #         else:
    #             word_doc_list[word] = [i]
    #         appeared.add(word)

    # word_doc_freq = {}
    # for word, doc_list in word_doc_list.items():
    #     word_doc_freq[word] = len(doc_list)

    # word_id_map = {}
    # id_word_map = {}
    # for i in range(vocab_size):
    #     word_id_map[vocab[i]] = i
    #     id_word_map[i] = vocab[i]

    # # ====================================================
    # # compute syntactic weights


    # # compute weights
    # for key in word_pair_count:
    #     temp = key.split(',')
    #     i = int(temp[0])
    #     j = int(temp[1])
    #     count = word_pair_count[key]
    #     word_freq_i = word_window_freq[vocab[i]]
    #     word_freq_j = word_window_freq[vocab[j]]
    #     pmi = log((1.0 * count / window_size) / (1.0 * word_freq_i * word_freq_j / (window_size * window_size)))
    #     if pmi <= 0:
    #         continue
    #     # pmi
    #     row.append(train_size + i)
    #     col.append(train_size + j)
    #     # weight.append(pmi)
    #     # Dependência sintática
    #     if i not in id_word_map or j not in id_word_map:
    #         continue
    #     newkey = id_word_map[i] + ',' + id_word_map[j]
    #     if newkey in rela_pair_count_str:
    #         # padronização min-max
    #         wei = (rela_pair_count_str[newkey] - min_count1) / (max_count1 - min_count1)
    #         # 0 normalização média
    #         # wei = (rela_pair_count_str[key]-count_mean1)/ count_std1
    #         # Taxa de frequência de ocorrência, mais frequentemente quando 1 aparece
    #         # wei = rela_pair_count_str[key] / data2[key]
    #         weight1.append(wei)
    #     else:
    #         weight1.append(pmi)

    # # ====================================================
    # # doc word frequency
    # weight_tfidf = []
    # doc_word_freq = {}
    # for doc_id in range(len(docs_of_words)):
    #     #doc_words = docs_of_words[doc_id]
    #     #words = doc_words.split()
    #     words = docs_of_words[doc_id]
    #     for word in words:
    #         word_id = word_id_map[word]
    #         doc_word_str = str(doc_id) + ',' + str(word_id)
    #         if doc_word_str in doc_word_freq:
    #             doc_word_freq[doc_word_str] += 1
    #         else:
    #             doc_word_freq[doc_word_str] = 1

    # for i in range(len(docs_of_words)):
    #     #doc_words = docs_of_words[i]
    #     #words = doc_words.split()
    #     words = docs_of_words[doc_id]
    #     doc_word_set = set()
    #     for word in words:
    #         if word in doc_word_set:
    #             continue
    #         j = word_id_map[word]
    #         key = str(i) + ',' + str(j)
    #         freq = doc_word_freq[key]
    #         if i < train_size:
    #             row.append(i)
    #         else:
    #             row.append(i + vocab_size)
    #         col.append(train_size + j)
    #         idf = log(1.0 * len(docs_of_words) / word_doc_freq[vocab[j]])
    #         weight_tfidf.append(freq * idf)
    #         doc_word_set.add(word)

    # weight = weight1 + weight_tfidf
    # node_size = train_size + vocab_size + test_size
    # adj = sp.csr_matrix((weight, (row, col)), shape=(node_size, node_size))

    # # dump objects
    # #ds_syntactic = cfg.corpus_shuffled_adjacency_dir + '/ind.{}.adj'.format(ds_name)
    # #f = open(ds_syntactic, 'wb')
    # #pkl.dump(adj, f)
    # # f.close()

    # # Dump Adjacency Matrix
    # with open(cfg.corpus_shuffled_adjacency_dir + "/syntactic/ind.{}.adj".format(ds_name), 'wb') as f:
    #     pkl.dump(adj, f)

    # elapsed = time() - t1
    # print("[INFO] Adjacency Dir='{}'".format(
    #     cfg.corpus_shuffled_adjacency_dir))
    # print("[INFO] Elapsed time is %f seconds." % elapsed)
    # print("[INFO] ========= EXTRACTED ADJACENCY MATRIX: Heterogenous doc-word adjacency matrix. =========")

import pickle
from time import time

from scipy.sparse import csr_matrix

from common import check_data_set
from preprocessors.configs import PreProcessingConfigs
from utils.file_ops import create_dir, check_paths
from utils.logger import PrintLog
import preprocessors.adjacency as adj


def build_graph_adjacency(ds_name: str, cfg: PreProcessingConfigs, pl: PrintLog):
    """Build Adjacency Matrix of Doc-Word Heterogeneous Graph"""

    t1 = time()
    #pl = PrintLog()
    # input files
    ds_corpus = cfg.corpus_shuffled_dir + ds_name + ".txt"
    ds_corpus_vocabulary = cfg.corpus_shuffled_vocab_dir + ds_name + '.vocab'
    ds_corpus_train_idx = cfg.corpus_shuffled_split_index_dir + ds_name + '.train'
    ds_corpus_test_idx = cfg.corpus_shuffled_split_index_dir + ds_name + '.test'

    # checkers
    check_data_set(data_set_name=ds_name, all_data_set_names=cfg.data_sets)
    check_paths(ds_corpus, ds_corpus_vocabulary,
                ds_corpus_train_idx, ds_corpus_test_idx)

    create_dir(dir_path=cfg.corpus_shuffled_adjacency_dir +
               "/graph", overwrite=False)

    docs_of_words = [line.split() for line in open(file=ds_corpus)]
    # Extract Vocabulary.
    vocab = open(ds_corpus_vocabulary).read().splitlines()
    word_to_id = {word: i for i, word in enumerate(vocab)}  # Word to its id.
    # Real train-size, not adjusted.
    train_size = len(open(ds_corpus_train_idx).readlines())
    test_size = len(open(ds_corpus_test_idx).readlines())  # Real test-size.

    windows_of_words = adj.extract_windows(
        docs_of_words=docs_of_words, window_size=20)

    # Extract word-word weights
    rows, cols, weights = adj.extract_pmi_word_weights(
        windows_of_words, word_to_id, vocab, train_size)
    # As an alternative, use cosine similarity of word vectors as weights:
    #   ds_corpus_word_vectors = cfg.CORPUS_WORD_VECTORS_DIR + ds_name + '.word_vectors'
    #   rows, cols, weights = extract_cosine_similarity_word_weights(vocab, train_size, ds_corpus_word_vectors)

    # Extract word-doc weights
    rows, cols, weights = adj.extract_tw_idf_doc_word_weights(
        rows, cols, weights, vocab, train_size, docs_of_words, word_to_id)

    adjacency_len = train_size + len(vocab) + test_size

    pl.print_log(
        f"[INFO] ({len(weights)}, ({len(rows)}, {len(cols)})), shape=({adjacency_len}, {adjacency_len})")

    adjacency_matrix = csr_matrix(
        (weights, (rows, cols)), shape=(adjacency_len, adjacency_len))

    # Dump Adjacency Matrix
    with open(cfg.corpus_shuffled_adjacency_dir + "/graph/ind.{}.adj".format(ds_name), 'wb') as f:
        pickle.dump(adjacency_matrix, f)

    elapsed = time() - t1
    pl.print_log("[INFO] Adjacency Dir='{}'".format(
        cfg.corpus_shuffled_adjacency_dir))
    pl.print_log("[INFO] Elapsed time is %f seconds." % elapsed)
    pl.print_log("[INFO] ========= EXTRACTED ADJACENCY MATRIX: Heterogenous doc-word adjacency matrix. =========")

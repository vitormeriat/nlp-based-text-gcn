from os.path import isabs
from os import getcwd


def make_path_absolute(a_path: str) -> str:
    if isabs(a_path):
        print('[WARN] Path:{} is already absolute.'.format(a_path))
        return a_path
    current_working_dir = getcwd()
    return f'{current_working_dir}/{a_path}'


class PreProcessingConfigs:
    def __init__(self):
        self.data_sets = None  # List of Valid Data-sets
        self.adjacency_sets = None  # List of Valid Adjacency-sets
        self.data_set_extension = None  # Extension of data-sets, e.g. "txt"
        self.corpus_dir = None  # Original Corpus Directory
        self.corpus_cleaned_dir = None  # Cleaned Corpus Directory
        self.corpus_meta_dir = None  # Original Meta Directory of Corpus
        self.corpus_shuffled_dir = None  # Shuffled Corpus Directory
        # Train and Test Index of Shuffled Corpus
        self.corpus_shuffled_split_index_dir = None
        self.corpus_shuffled_meta_dir = None  # Meta Directory of Shuffled Corpus
        self.corpus_shuffled_vocab_dir = None  # Vocabulary of Shuffled Corpus
        self.corpus_shuffled_word_vectors_dir = None  # Word-Vectors of Shuffled Corpus
        # Node Features (x,y,tx,ty,allx) of Shuffled Corpus
        self.corpus_shuffled_node_features_dir = None
        # Adjacency Matrix (adj) of Shuffled Corpus
        self.corpus_shuffled_adjacency_dir = None
        self.core_nlp_path = None
        self.liwc_path = None

    def build(self) -> 'PreProcessingConfigs':
        self.corpus_dir = make_path_absolute(self.corpus_dir)
        self.corpus_cleaned_dir = make_path_absolute(self.corpus_cleaned_dir)
        self.corpus_meta_dir = make_path_absolute(self.corpus_meta_dir)
        self.corpus_shuffled_dir = make_path_absolute(self.corpus_shuffled_dir)
        self.corpus_shuffled_split_index_dir = make_path_absolute(
            self.corpus_shuffled_split_index_dir)
        self.corpus_shuffled_meta_dir = make_path_absolute(
            self.corpus_shuffled_meta_dir)
        self.corpus_shuffled_vocab_dir = make_path_absolute(
            self.corpus_shuffled_vocab_dir)
        self.corpus_shuffled_word_vectors_dir = make_path_absolute(
            self.corpus_shuffled_word_vectors_dir)
        self.corpus_shuffled_node_features_dir = make_path_absolute(
            self.corpus_shuffled_node_features_dir)
        self.corpus_shuffled_adjacency_dir = make_path_absolute(
            self.corpus_shuffled_adjacency_dir)
        self.core_nlp_path = make_path_absolute(self.core_nlp_path)
        return self

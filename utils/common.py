from typing import Any, Dict, Iterable, Optional
from collections import Counter
from nltk import word_tokenize
from typing import List


def tokenize_prune_stem(text: str, stemming_map: Optional[Dict[str, str]] = None) -> List[str]:
    """
    Use NLTK word tokenisation and clean our text, and (if passed to the function) use stemming map to stem all words.
    NOTE: If `word` is not in `stemming_map`, we do not include it
    """
    return [
        stemming_map[word] if stemming_map is not None else word
        for word in word_tokenize(text)
        # if len(word) > 1 and word.isalpha() and word not in STOP_WORDS and word in stemming_map
        if len(word) > 1 and word.isalpha() and word in stemming_map
    ]


def extract_word_counts(docs_of_words: List[List[str]]) -> Counter:
    """Extract word counts"""
    word_counts = Counter()
    for words in docs_of_words:
        word_counts.update(words)
    return word_counts


def check_data_set(data_set_name: str, all_data_set_names: List[str]) -> None:
    if data_set_name not in all_data_set_names:
        raise AttributeError("Wrong data-set name, given:%r, however expected:%r" %
                             (data_set_name, all_data_set_names))


def flatten_nested_iterables(iterables_of_iterables: Iterable[Iterable[Any]]) -> Iterable[Any]:
    return [item for sublist in iterables_of_iterables for item in sublist]


def get_hyperparameters():
    hyperparameters = []
    with open('hyperparameters.csv', 'r') as f:
        lines = f.readlines()[1:]
        for l in lines:
            h = l.replace('\n', '').split(';')
            hyperparameters.append({
                'experiment': int(h[0]),
                'learning_rate': float(h[1]),
                'hidden_1': int(h[2]),
                'dropout': float(h[3]),
                'early_stopping': int(h[4]),
                'epochs': int(h[5]),
                'weight_decay': int(h[6]),
                'max_degree': int(h[7]),
                'model': h[8]
            })

    return hyperparameters

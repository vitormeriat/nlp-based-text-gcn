# GCN for text classification

GCN applied in a text classification context.

## Abstract
This project aims to exam the text classification problem with novel approaches Graph Convolutional Networks and Graph Attention Networks using Deep Learning algorithms and Natural Language Processing Techniques.


### **Available Datasets:**

+ 20ng (Newsgroup Dataset)
+ R8 (Reuters News Dataset with 8 labels)
+ R52 (Reuters News Dataset with 52 labels)
+ Ohsumed (Cardiovascular Diseases Abstracts Dataset)
+ MR (Movie Reviews Dataset)
+ Cora (Citation Dataset)
+ Citeseer (Citation Dataset)
+ Pubmed (Citation Dataset)

### **Datasets used in experiments:**

| Dataset | Docs | Training | Test | Words | Nodes | Classes | Average Length |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 20NG    | 18,846 | 11,314 | 7,532 | 42,757 | 61,603 | 20 | 221.26 |
| R8      | 7,674  | 5,485  | 2,189 | 7,688  | 15,362 | 8  | 65.72  |
| R52     | 9,100  | 6,532  | 2,568 | 8,892  | 17,992 | 52 | 69.82  |
| MR      | 10,662 | 7,108  | 3,554 | 18,764 | 29,426 | 2  | 20.39  |
| Ohsumed | 7,400  | 3,357  | 4,043 | 14,157 | 21,557 | 23 | 135.82 |

### **Available Text Model Representations:**

| MODEL| COMMAND | DESCRIPTION |
| --- | --- | --- |
| Frequency | `frequency` | TF-IDF / PMI |
| Syntactic Dependency Tree | `syntactic_dependency` | --- |
| LIWC | `linguistic_inquiry` | LIWC |
| Semantic | `semantic` | Word2Vec / Doc2Vec |
| Meaningful Term Weights | `graph` | TW-IDF  / PMI |

### **Preprocess:**

```bash
preprocess.py <TEXT_MODEL_REPRESENTATION>
```

*Example:* ```python3 preprocess.py linguistic_inquiry```

### **Train:**

```bash
train.py <TEXT_MODEL_REPRESENTATION>
```

*Example:* ```python3 train.py syntactic_dependency```

# References

## Papers 
+ [Kipf and Welling, 2017]  Semi-supervised Classification with Graph Convolutional Networks
+ [Liang Yao, Chengsheng Mao, Yuan Luo, 2018] Graph Convolutional Networks for Text Classification

## Books

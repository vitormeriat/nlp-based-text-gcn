# GCN for text classification

GCN applied in a text classification context.

## Abstract
This project aims to exam the text classification problem with novel approaches Graph Convolutional Networks and Graph Attention Networks using Deep Learning algorithms and Natural Language Processing Techniques.

#### **Available Datasets:**

+ 20ng (Newsgroup Dataset)
+ R8 (Reuters News Dataset with 8 labels)
+ R52 (Reuters News Dataset with 52 labels)
+ ohsumed (Cardiovascular Diseases Abstracts Dataset)
+ mr (Movie Reviews Dataset)
+ cora (Citation Dataset)
+ citeseer (Citation Dataset)
+ pubmed (Citation Dataset)

**Preprocess:**
```bash
preprocess.py <DATASET_NAME>
```
*Example:* ```python3 preprocess.py R8```

**Train:**
```bash
train.py <DATASET_NAME>
```
*Example:* ```python3 train.py R8```

## References

### Papers 
+ [Kipf and Welling, 2017]  Semi-supervised Classification with Graph Convolutional Networks
+ [Liang Yao, Chengsheng Mao, Yuan Luo, 2018] Graph Convolutional Networks for Text Classification

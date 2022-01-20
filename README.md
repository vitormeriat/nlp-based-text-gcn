# GCN for text classification

<div id="top"></div>

GCN applied in a text classification context.

### **Table of Contents**

<ol>
    <li><a href="#abstract">Abstract</a></li>
    <li><a href="#installation">Installation</a></li>
    <li>
        <a href="#datasets">Datasets</a>
        <ul>
            <li><a href="#available-datasets">Available Datasets</a></li>
            <li><a href="#datasets-description">Datasets Description</a></li>
        </ul>
    </li>
    <li><a href="#train">Train</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#citation">Citation</a></li>
    <li><a href="#references">References</a></li>
</ol>


<div style="margin-bottom:30px"></div>

## Abstract

This project aims to exam the text classification problem with novel approaches Graph Convolutional Networks and Graph Attention Networks using Deep Learning algorithms and Natural Language Processing Techniques.

<p style="margin-bottom:30px" align="right">(<a href="#top">back to top</a>)</p>

## Datasets

### **Available Datasets:**

+ 20ng (Newsgroup Dataset)
+ R8 (Reuters News Dataset with 8 labels)
+ R52 (Reuters News Dataset with 52 labels)
+ Ohsumed (Cardiovascular Diseases Abstracts Dataset)
+ MR (Movie Reviews Dataset)
+ Cora (Citation Dataset)
+ Citeseer (Citation Dataset)
+ Pubmed (Citation Dataset)

### **Datasets Description:**

| Dataset | Docs | Training | Test | Words | Nodes | Classes | Average Length |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 20NG    | 18,846 | 11,314 | 7,532 | 42,757 | 61,603 | 20 | 221.26 |
| R8      | 7,674  | 5,485  | 2,189 | 7,688  | 15,362 | 8  | 65.72  |
| R52     | 9,100  | 6,532  | 2,568 | 8,892  | 17,992 | 52 | 69.82  |
| MR      | 10,662 | 7,108  | 3,554 | 18,764 | 29,426 | 2  | 20.39  |
| Ohsumed | 7,400  | 3,357  | 4,043 | 14,157 | 21,557 | 23 | 135.82 |

<p style="margin-bottom:30px" align="right">(<a href="#top">back to top</a>)</p>

## Train


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

<p style="margin-bottom:30px" align="right">(<a href="#top">back to top</a>)</p>


## Contributing

Contributions are **greatly appreciated**. If you want to help us improve this software, please fork the repo and create a new pull request. Don't forget to give the project a star! Thanks!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Alternatively, you can make suggestions or report bugs by opening a new issue with the appropriate tag 
("feature" or "bug") and following our Contributing template.

<p style="margin-bottom:30px" align="right">(<a href="#top">back to top</a>)</p>


## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p style="margin-bottom:30px" align="right">(<a href="#top">back to top</a>)</p>


## Citation

If you have found TGCN useful in your research, please consider giving the repo a star and citing our [arXiv paper]():

```bibtex
    @misc{Vitor2022,
      author = {},
      title = {},
      year = {2022},
      archivePrefix = {},
      eprint = {}
    }
```

<p style="margin-bottom:30px" align="right">(<a href="#top">back to top</a>)</p>


## References

### **Papers** 

+ [Kipf and Welling, 2017]  Semi-supervised Classification with Graph Convolutional Networks
+ [Liang Yao, Chengsheng Mao, Yuan Luo, 2018] Graph Convolutional Networks for Text Classification

### **Books**

+ 

<p align="right">(<a href="#top">back to top</a>)</p>
<img src="docs/img/icon.png" width=125 height=125 align="right">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-2108.11792-b31b1b.svg?style=plastic)](https://arxiv.org/abs/2108.11792)
[![Loader: Hugging Face Datasets](https://img.shields.io/static/v1.svg?label=🤗%20Datasets&message=BSARD&color=FF9900)](https://huggingface.co/datasets/maastrichtlawtech/bsard)

# A Statutory Article Retrieval Dataset in French

This repository contains the code for reproducing the experimental results presented in the ACL 2022 paper ["A Statutory Article Retrieval Dataset in French"](https://aclanthology.org/2022.acl-long.468/) by [Antoine Louis](https:/antoinelouis.co/work/) and [Jerry Spanakis](https://dke.maastrichtuniversity.nl/jerry.spanakis/).

<img align="center" src="docs/img/task.png" width="1000">

Statutory article retrieval is the task of automatically retrieving law articles relevant to a legal question. While recent advances in natural language processing have sparked considerable interest in many legal tasks, statutory article retrieval remains primarily untouched due to the scarcity of large-scale and high-quality annotated datasets. To address this bottleneck, we introduce the Belgian Statutory Article Retrieval Dataset (BSARD), which consists of 1,100+ French native legal questions labeled by experienced jurists with relevant articles from a corpus of 22,600+ Belgian law articles. Using BSARD, we benchmark several state-of-the-art retrieval approaches, including lexical and dense architectures, both in zero-shot and supervised setups. We find that fine-tuned dense retrieval models significantly outperform other systems. Our best performing baseline achieves 74.8% R@100, which is promising for the feasibility of the task and indicates there is still room for improvement. By the specificity of the domain and addressed task, BSARD presents a unique challenge problem for future research on legal information retrieval.

## Documentation

Detailed documentation on the dataset and how to reproduce the main experimental results can be found [here](docs/README.md).

## Citation

For attribution in academic contexts, please cite this work as:

```latex
@inproceedings{louis2022statutory,
  title = {A Statutory Article Retrieval Dataset in French},
  author = {Louis, Antoine and Spanakis, Gerasimos},
  booktitle = {Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics},
  month = may,
  year = {2022},
  address = {Dublin, Ireland},
  publisher = {Association for Computational Linguistics},
  url = {https://aclanthology.org/2022.acl-long.468},
  pages = {6789--6803},
}
```

## License

This repository is MIT-licensed.

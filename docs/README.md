<img src="img/icon.png" width=125 height=125 align="right">

# A Statutory Article Retrieval Dataset in French

This repository contains:

* The Belgian Statutory Article Retrieval Dataset (BSARD)  v1.0.
* Code for hyper-parameters search and evaluation of the retriever models.
* Web application to visualize insightful statistics about BSARD.

## Belgian Statutory Article Retrieval Dataset (BSARD)

We provide the *dataset nutrition labels* [(Holland et al., 2018)](https://arxiv.org/abs/1805.03677) for BSARD.

<p align="center"><img align="center" src="img/nutrition.png" width="70%"></p>

## Experiments

### Setup

This repository is tested on Python 3.8+. To install all dependencies, you should have [conda](https://docs.conda.io/projects/conda/en/latest/index.html) installed on your machine and run:

```bash
conda env create -f environment.yml
conda activate bsard
```

In addition, please install spaCy's [fr_core_news_md](https://spacy.io/models/fr#fr_core_news_md) pipeline (needed for text processing) by running:

```bash
python -m spacy download fr_core_news_md
```

Lastly, download the pre-trained French [fastText](https://fasttext.cc/docs/en/crawl-vectors.html#models) and [word2vec](https://fauconnier.github.io/#data) embeddings by running:

```bash
bash scripts/download_embeddings.sh
```

### Search hyper-parameters

In order to find the optimal hyper-parameters for one type of retriever model (i.e., one of {word2vec, fasttext, bert}), run:

```bash
python scripts/search_hyperparameters.py \
    --articles_path </path/to/articles.csv> \
    --questions_path </path/to/questions_train.csv> \
    --retriever_model {word2vec, fasttext, bert} \ 
    --embeddings_dir </path/to/downloaded/embbedings> \  # [Only for word2vec/fasttext]
    --lem \                                              # [Optional] Lemmatize both articles and questions during pre-processing.
    --output_dir </path/to/output>
```

This script will evaluate all possible combinations between the following parameters:

* **Retriever model**:
  * For [word2vec](https://fauconnier.github.io/#data): {*CBOW, skip-gram*} model pre-trained on {*frWiki, frWac, frWiki-lemmatized, frWac-lemmatized*} with embedding size of {*200, 500, 700, 1000*}
  * For [fastText](https://fasttext.cc/docs/en/crawl-vectors.html): *CBOW* model pre-trained on French webpages with embedding size of *300*
  * For BERT: {*CamemBERT, FlauBERT, mBERT, DistilmBERT*}
* **Pooling operation**: {*mean, max, sum*}
* **Distance function**: {*euclidean, cosine*}

### Test

In order to test the retriever models on the test set, run:

```bash
python scripts/test.py \
    --articles_path </path/to/articles.csv> \
    --questions_path </path/to/questions_test.csv> \
    --retriever_model {tfidf, bm25, word2vec, fasttext, bert} \ 
    --lem \                                                     # [Optional] Lemmatize both articles and questions during pre-processing.
    --output_dir </path/to/output>
```

### Visualize

We provide a [Dash](https://plotly.com/dash/) web application that shows insightful visualizations about BSARD.

<p align="center"><img src="img/eda.gif" width="80%" height="auto"></p>

To explore the visualizations on your local machine, run:

```bash
python scripts/eda/visualise.py
```

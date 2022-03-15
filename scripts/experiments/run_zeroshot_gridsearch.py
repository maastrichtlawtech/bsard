import os
import ntpath
import pickle
import itertools
import argparse
from os.path import abspath, join, splitext

import numpy as np
import pandas as pd

from utils.eval import Evaluator
from utils.data import TextPreprocessor
from models.lexical_models import BM25Retriever
from models.zeroshot_dense_models import Word2vecRetriever, FasttextRetriever, BERTRetriever



def run_gridsearch_bm25(articles, questions, ground_truths, topk, outdir):
    # Init evaluator.
    evaluator = Evaluator()

    # Create dataframe to store results.
    hyperparameters = ['k1', 'b']
    metrics = [f"{key}@{v}" for key, values in evaluator.metrics_at_k.items() for v in values]
    grid_df = pd.DataFrame(columns=hyperparameters+metrics)

    # Create all possible combinations of hyperparamaters.
    k1_range = np.arange(0, 8.5, 0.5)
    b_range = np.arange(0, 1.1, 0.1)
    combinations = list(itertools.product(*[k1_range, b_range]))

    # Launch grid search runs.
    for i, (k1, b) in enumerate(combinations):
        print(f"\n\n({i+1}) Model: BM25 - k1={k1}, b={b}")
        retriever = BM25Retriever(retrieval_corpus=articles, k1=k1, b=b)
        retrieved_docs = retriever.search_all(questions, top_k=topk)
        scores = evaluator.compute_all_metrics(retrieved_docs, ground_truths)

        scores.update({'k1':k1, 'b':b})
        grid_df = grid_df.append(scores, ignore_index=True)
        grid_df.to_csv(join(outdir, 'bm25_results.csv'), sep=',', float_format='%.5f', index=False)
    
    return grid_df


def run_gridsearch_dense_models(retriever, checkpoints, articles, questions, ground_truths, topk, outdir):
    # Init evaluator.
    evaluator = Evaluator()

    # Create dataframe to store results.
    hyperparameters = ['model_name', 'pooling_mode', 'distance_fn']
    metrics = [f"{key}@{v}" for key, values in evaluator.metrics_at_k.items() for v in values]
    grid_df = pd.DataFrame(columns=hyperparameters+metrics)

    # Create all possible combinations of hyperparamaters.
    pooling = ['mean', 'max', 'sum']
    combinations = list(itertools.product(*[checkpoints, pooling]))

    # Launch grid search runs.
    for i, (model, pooling) in enumerate(combinations):
        print(f"\n\n({i+1}) Model: {model} - Pooling: {pooling}")
        # Embbed all articles once with given retriever and pooling strat.
        ret = retriever(model_path_or_name=model, pooling_strategy=pooling, retrieval_corpus=articles)
        for dist in ['cosine', 'euclidean']:
            print(f"\n - Distance: {dist}")
            retrieved_docs = ret.search_all(questions, top_k=topk, dist_metric=dist)
            scores = evaluator.compute_all_metrics(retrieved_docs, ground_truths)

            # Save scores to dataframe.
            model_name = splitext(ntpath.basename(model))[0]
            scores.update({'model_name':model_name, 'pooling_mode':pooling, 'distance_fn':dist})
            grid_df = grid_df.append(scores, ignore_index=True)
            grid_df.to_csv(join(outdir, f'{repr(ret)}_results.csv'), sep=',', float_format='%.5f', index=False)

    return grid_df


def get_checkpoint_paths(checkpoints_dir):
    assert checkpoints_dir is not None, "Please set --models_dir (path of directory containing the embeddings) when using word2vec or fasttext."
    _, _, models = next(os.walk(checkpoints_dir))
    filepaths = [join(checkpoints_dir, model) for model in models if model.endswith(".bin")]
    return filepaths


def main(args):
    os.makedirs(args.outdir, exist_ok=True)

    print("Loading questions, article, and ground truth labels...")
    dfA = pd.read_csv(args.articles_path)
    dfQ = pd.read_csv(args.questions_path)
    ground_truths = dfQ['article_ids'].apply(lambda x: list(map(int, x.split(',')))).tolist()

    if not args.retriever == 'bert':
        print(f"Preprocessing articles and questions (lemmatizing={args.lem})...")
        cleaner = TextPreprocessor(spacy_model="fr_core_news_md")
        articles = cleaner.preprocess(dfA['article'], lemmatize=args.lem)
        questions = cleaner.preprocess(dfQ['question'], lemmatize=args.lem)
    else:
        articles = dfA['article'].tolist()
        questions = dfQ['question'].tolist()

    print("Running gridsearch...")
    if args.retriever == "bm25":
        results = run_gridsearch_bm25(
            articles=articles, 
            questions=questions, 
            ground_truths=ground_truths, 
            topk=500, 
            outdir=args.outdir,
        )
    else:
        if args.retriever == "word2vec":
            models = get_checkpoint_paths(args.models_dir)
            retriever = Word2vecRetriever
        elif args.retriever == "fasttext":
            models = get_checkpoint_paths(args.models_dir)
            retriever = FasttextRetriever
        elif args.retriever == "bert":
            models = ['camembert-base', 'flaubert/flaubert_base_cased', 'flaubert/flaubert_small_cased']
            retriever = BERTRetriever
        results = run_gridsearch_dense_models(
            retriever=retriever, 
            checkpoints=models, 
            articles=articles, 
            questions=questions, 
            ground_truths=ground_truths,
            topk=500, 
            outdir=args.outdir,
        )



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--articles_path", 
                        type=str, 
                        default=abspath(join(__file__, "../../../data/bsard_v1/articles_fr.csv")),
                        help="Path of the data file containing the law articles."
    )
    parser.add_argument("--questions_path", 
                        type=str, 
                        default=abspath(join(__file__, "../../../data/bsard_v1/questions_fr_train.csv")),
                        help="Path of the data file containing the test questions."
    )
    parser.add_argument("--lem",
                        action='store_true', 
                        default=False,
                        help="Lemmatize the questions and articles for retrieval."
    )
    parser.add_argument("--retriever", 
                        type=str,
                        choices=["bm25","word2vec","fasttext","bert"],
                        required=True,
                        help="The type of model to use for retrieval"
    )
    parser.add_argument("--models_dir", 
                        type=str,
                        default=abspath(join(__file__, "../embeddings/")),
                        help="Path of the directory containing the embedding models (only needed for word2vec and fasttext)."
    )
    parser.add_argument("--outdir",
                        type=str, 
                        default=abspath(join(__file__, "../output/zeroshot/gridsearch/")),
                        help="Path of the output directory."
    )
    args, _ = parser.parse_known_args()
    main(args)

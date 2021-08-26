import os
import json
import pickle
import argparse
import pandas as pd
from gensim.models import KeyedVectors

from evaluator import Evaluator
from processor import TextPreprocessor
from retriever import (TFIDFRetriever, BM25Retriever, SWSNRetriever,
                       Word2vecRetriever, FasttextRetriever, BERTRetriever)


def main(args):
    print("Loading questions and articles...")
    dfA = pd.read_csv(args.articles_path)
    dfQ = pd.read_csv(args.questions_path)

    if not args.retriever_model == 'bert':
        print("Preprocessing articles and questions (lemmatizing={})...".format(args.lem))
        cleaner = TextPreprocessor(spacy_model="fr_core_news_md")
        articles = cleaner.preprocess(dfA['article'], lemmatize=args.lem)
        questions = cleaner.preprocess(dfQ['question'], lemmatize=args.lem)
    else:
        articles = dfA['article'].tolist()
        questions = dfQ['question'].tolist()

    print("Initializing the {} retriever model...".format(args.retriever_model))
    if args.retriever_model == 'tfidf':
        retriever = TFIDFRetriever(retrieval_corpus=articles)
    elif args.retriever_model == 'bm25':
        retriever = BM25Retriever(retrieval_corpus=articles, k1=1.2, b=0.75)
    elif args.retriever_model == 'word2vec':
        retriever = Word2vecRetriever(model_path_or_name='models/word2vec/lemmatized/word2vec_frWac_lem_skipgram_d500.bin', pooling_strategy='mean', retrieval_corpus=articles)
    elif args.retriever_model == 'fasttext':
        retriever = FasttextRetriever(model_path_or_name='models/fasttext/fasttext_frCC_cbow_d300.bin', pooling_strategy='mean', retrieval_corpus=articles)
    elif args.retriever_model == 'bert':
        retriever = BERTRetriever(model_path_or_name='distilmbert', pooling_strategy='max', retrieval_corpus=articles)

    print("Running model on test questions...")
    if args.retriever_model == 'tfidf' or args.retriever_model == 'bm25':
        results = retriever.search_all(questions, top_k=100)
    else:
        results = retriever.search_all(questions, top_k=100, dist_metric='cosine')

    print("Computing the scores...")
    evaluator = Evaluator()
    ground_truths = dfQ['article_ids'].apply(lambda x: list(map(int, x.split(',')))).tolist()
    scores = evaluator.compute_all_metrics(ground_truths, results)

    print("Saving the results and scores to {} ...".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(os.path.join(args.output_dir, '{}'.format(args.retriever_model) + '.pkl'), 'wb') as f:
        pickle.dump(results, f)
    with open(os.path.join(args.output_dir, '{}'.format(args.retriever_model) + '_scores.json'), 'w', encoding='utf-8') as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)
    print("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--articles_path", 
                        type=str, 
                        default="../data/bsard_v1/articles_fr.csv",
                        help="Path of the data file containing the law articles."
    )
    parser.add_argument("--questions_path", 
                        type=str, 
                        default="../data/bsard_v1/questions_fr_test.csv",
                        help="Path of the data file containing the testing questions."
    )
    parser.add_argument("--lem",
                        action='store_true', 
                        default=False,
                        help="Lemmatize the questions and articles for retrieval."
    )
    parser.add_argument("--retriever_model", 
                        type=str,
                        choices  = ["tfidf","bm25","word2vec","fasttext","bert"],
                        required=True,
                        help="The type of model to use for retrieval"
    )
    parser.add_argument("--output_dir",
                        type=str, 
                        default="./output/test/",
                        help="Path of the output directory."
    )
    args, _ = parser.parse_known_args()
    main(args)

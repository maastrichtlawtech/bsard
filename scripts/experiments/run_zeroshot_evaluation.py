import os
import json
import argparse
from os.path import abspath, join

import pandas as pd
from gensim.models import KeyedVectors

from utils.eval import Evaluator
from utils.data import TextPreprocessor
from models.lexical_models import TFIDFRetriever, BM25Retriever, SWSNRetriever
from models.zeroshot_dense_models import Word2vecRetriever, FasttextRetriever, BERTRetriever



def main(args):
    print("Loading questions and articles...")
    dfA = pd.read_csv(args.articles_path)
    dfQ_test = pd.read_csv(args.test_questions_path)
    ground_truths = dfQ_test['article_ids'].apply(lambda x: list(map(int, x.split(',')))).tolist()

    if not args.retriever == 'bert':
        print("Preprocessing articles and questions (lemmatizing={})...".format(args.lem))
        cleaner = TextPreprocessor(spacy_model="fr_core_news_md")
        articles = cleaner.preprocess(dfA['article'], lemmatize=args.lem)
        questions = cleaner.preprocess(dfQ_test['question'], lemmatize=args.lem)
    else:
        articles = dfA['article'].tolist()
        questions = dfQ_test['question'].tolist()

    print("Initializing the {} retriever model...".format(args.retriever))
    if args.retriever == 'tfidf':
        retriever = TFIDFRetriever(retrieval_corpus=articles)
    elif args.retriever == 'bm25':
        retriever = BM25Retriever(retrieval_corpus=articles, k1=1.0, b=0.6)
    elif args.retriever == 'word2vec':
        best_checkpoint = abspath(join(__file__, "../embeddings/word2vec/lemmatized/word2vec_frWac_lem_skipgram_d500.bin"))
        retriever = Word2vecRetriever(model_path_or_name=best_checkpoint, pooling_strategy='mean', retrieval_corpus=articles)
    elif args.retriever == 'fasttext':
        best_checkpoint = abspath(join(__file__, "../embeddings/fasttext/fasttext_frCc_cbow_d300.bin"))
        retriever = FasttextRetriever(model_path_or_name=best_checkpoint, pooling_strategy='mean', retrieval_corpus=articles)
    elif args.retriever == 'bert':
        retriever = BERTRetriever(model_path_or_name='camembert-base', pooling_strategy='mean', retrieval_corpus=articles)

    print("Running model on test questions...")
    if args.retriever == 'tfidf' or args.retriever == 'bm25':
        retrieved_docs = retriever.search_all(questions, top_k=500)
    else:
        retrieved_docs = retriever.search_all(questions, top_k=500, dist_metric='cosine')

    print("Computing the retrieval scores...")
    evaluator = Evaluator()
    scores = evaluator.compute_all_metrics(retrieved_docs, ground_truths)
    
    print("Saving the scores to {} ...".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)
    with open(join(args.output_dir, f'{args.retriever}_test_results.json'), 'w') as f:
        json.dump(scores, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--articles_path", 
                        type=str, 
                        default=abspath(join(__file__, "../../../data/bsard_v1/articles_fr.csv")),
                        help="Path of the data file containing the law articles."
    )
    parser.add_argument("--test_questions_path", 
                        type=str, 
                        default=abspath(join(__file__, "../../../data/bsard_v1/questions_fr_test.csv")),
                        help="Path of the data file containing the test questions."
    )
    parser.add_argument("--lem",
                        action='store_true', 
                        default=False,
                        help="Lemmatize the questions and articles for retrieval."
    )
    parser.add_argument("--retriever", 
                        type=str,
                        choices=["tfidf","bm25","word2vec","fasttext","bert"],
                        required=True,
                        help="The type of model to use for retrieval"
    )
    parser.add_argument("--output_dir",
                        type=str, 
                        default=abspath(join(__file__, "../output/zeroshot/test-run/")),
                        help="Path of the output directory."
    )
    args, _ = parser.parse_known_args()
    main(args)

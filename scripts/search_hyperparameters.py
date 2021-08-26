import os
import json
import ntpath
import pickle
import itertools
import argparse
import pandas as pd

from evaluator import Evaluator
from processor import TextPreprocessor
from retriever import Word2vecRetriever, FasttextRetriever, BERTRetriever


def run_gridsearch(articles, questions, retriever, models, ground_truths, outdir):
    # Create all combinations between models and pooling strategies.
    combinations = list(itertools.product(*[models, ['mean', 'max', 'sum']]))

    print("Running Grid Search...")
    evaluator = Evaluator()
    for i, (model, pooling) in enumerate(combinations):
        print("\n\n({}) Model: {} - Pooling: {}".format(i+1, model, pooling))
        # Init a retriever model by embedding all articles from retrieval corpus.
        ret = retriever(model_path_or_name=model, pooling_strategy=pooling, retrieval_corpus=articles)

        for dist in ['cosine', 'euclidean']:
            print("\n - Distance: {}".format(dist))
            # Get search results.
            results = ret.search_all(questions, top_k=100, dist_metric=dist)

            # Compute metrics.
            scores = evaluator.compute_all_metrics(ground_truths, results)

            # Save results and scores.
            model_type = os.path.splitext(ntpath.basename(model))[0]
            results_outdir = os.path.join(outdir, model_type)
            if not os.path.exists(results_outdir):
                os.makedirs(results_outdir)

            model_name = '_'.join([model_type, pooling, dist])
            with open(os.path.join(results_outdir, model_name + '.pkl'), 'wb') as f:
                pickle.dump(results, f)
            with open(os.path.join(results_outdir, model_name + '_scores.json'), 'w', encoding='utf-8') as f:
                json.dump(scores, f, ensure_ascii=False, indent=4)
            print("Done.")


def main(args):
    # Loading questions and articles.
    dfA = pd.read_csv(args.articles_path)
    dfQ = pd.read_csv(args.questions_path)
    ground_truths = dfQ['article_ids'].apply(lambda x: list(map(int, x.split(',')))).tolist()

    # Extracting articles and questions.
    if args.retriever_model == 'word2vec' or args.retriever_model == 'fasttext':
        print("Preprocessing articles and questions (lemmatizing={})...".format(args.lem))
        cleaner = TextPreprocessor(spacy_model="fr_core_news_md")
        articles = cleaner.preprocess(dfA['article'], lemmatize=args.lem)
        questions = cleaner.preprocess(dfQ['question'], lemmatize=args.lem)
    else:
        articles = dfA['article'].tolist()
        questions = dfQ['question'].tolist()

    # Extracting list of models.
    if args.retriever_model == 'word2vec' or args.retriever_model == 'fasttext':
        _, _, models = next(os.walk(args.embeddings_dir))
        models = [os.path.join(args.embeddings_dir, model) for model in models if model.endswith(".bin")]
    elif args.retriever_model == 'bert':
        models = ['camembert', 'flaubert', 'mbert', 'distilmbert']

    # Run gridsearch.
    if args.retriever_model == 'word2vec':
        retriever = Word2vecRetriever
    elif args.retriever_model == 'fasttext':
        retriever = FasttextRetriever
    elif args.retriever_model == 'bert':
        retriever = BERTRetriever
    run_gridsearch(articles=articles, questions=questions, 
                   retriever=retriever, models=models,
                   ground_truths=ground_truths, outdir=args.output_dir)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--articles_path", 
                        type=str, 
                        default="../data/bsard_v1/articles_fr.csv",
                        help="Path of the data file containing the law articles."
    )
    parser.add_argument("--questions_path", 
                        type=str, 
                        default="../data/bsard_v1/questions_fr_train.csv",
                        help="Path of the data file containing the training questions."
    )
    parser.add_argument("--lem",
                        action='store_true', 
                        default=False,
                        help="Lemmatize the questions and articles during pre-processing."
    )
    parser.add_argument("--retriever_model", 
                        type=str,
                        choices  = ["word2vec","fasttext","bert"],
                        required=True,
                        help="The type of model to use for retrieval"
    )
    parser.add_argument("--embeddings_dir", 
                        type=str, 
                        help="Path of the directory containing the pre-trained embeddings (only needed for word2vec and fasttext)."
    )
    parser.add_argument("--output_dir",
                        type=str, 
                        default="output/train/",
                        help="Path of the output directory."
    )
    args, _ = parser.parse_known_args()
    main(args)

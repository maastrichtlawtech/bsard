import sys
import logging
import itertools
from tqdm import tqdm

import math
import random
import numpy as np
from statistics import mean
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity

import torch
import fasttext
from gensim.models import KeyedVectors
from transformers import (CamembertModel, CamembertTokenizer,
                          FlaubertModel, FlaubertTokenizer,
                          BertTokenizer, BertModel,
                          DistilBertTokenizer, DistilBertModel)

fasttext.FastText.eprint = lambda x: None
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)


#-----------------------------------#
#       BAG-OF-WORDS RETRIEVERS     #
#-----------------------------------#
class TFIDFRetriever:
    def __init__(self, retrieval_corpus):
        self.retrieval_corpus = retrieval_corpus
        self.N = len(retrieval_corpus)
        self.vocab = self._build_vocabulary()
        self.idfs = self._compute_idfs()

    def search_all(self, queries, top_k):
        results = list()
        for q in tqdm(queries, desc='Searching queries'):
            results.append([doc_id for doc_id,_ in self.search(q, top_k)])
        return results

    def search(self, q, top_k):
        results = dict()
        for i, doc in enumerate(self.retrieval_corpus):
            results[i+1] = self.score(q, doc)
        return sorted(results.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def score(self, q, d):
        score = 0.0
        for t in q.split():
            score += self._compute_tfidf(t, d)
        return score

    def _build_vocabulary(self):
        return sorted(set(itertools.chain.from_iterable([doc.lower().split() for doc in self.retrieval_corpus])))

    def _compute_idfs(self):
        idfs = dict.fromkeys(self.vocab, 0)
        for word,_ in idfs.items():
            idfs[word] = self._compute_idf(word)
        return idfs

    def _compute_idf(self, t):
        df = sum([1 if t in doc else 0 for doc in self.retrieval_corpus])
        return math.log10(self.N / (df + 1))

    def _compute_tf(self, t, d):
        return d.split().count(t)

    def _compute_tfidf(self, t, d):
        tf = self._compute_tf(t, d)
        idf = self.idfs[t] if t in self.idfs else math.log10(self.N)
        return tf * idf


class BM25Retriever(TFIDFRetriever):
    def __init__(self, retrieval_corpus, k1, b):
        super().__init__(retrieval_corpus)
        self.k1 = k1
        self.b = b
        self.avgdl = self._compute_avgdl()
    
    def score(self, q, d):
        score = 0.0
        for t in q.split():
            tf = self._compute_tf(t, d)
            idf = self.idfs[t] if t in self.idfs else math.log10((self.N + 0.5)/0.5)
            score += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * len(d.split())/self.avgdl))
        return score

    def _compute_avgdl(self):
        return mean([len(doc.split()) for doc in self.retrieval_corpus])

    def _compute_idf(self, t):
        df = sum([1 if t in doc else 0 for doc in self.retrieval_corpus])
        return math.log10((self.N - df + 0.5) / (df + 0.5))


class SWSNRetriever(BM25Retriever):
    def __init__(self, retrieval_corpus, k1, b, model):
        super().__init__(retrieval_corpus, k1, b)
        self.model = model

    def score(self, q, d):
        score = 0.0
        for t in d.split():
            sem = self._compute_sem(t, q)
            idf = self.idfs[t] if t in self.idfs else math.log10((self.N + 0.5)/0.5)
            score += idf * (sem * (self.k1 + 1)) / (sem + self.k1 * (1 - self.b + self.b * len(q.split())/self.avgdl))
        return score

    def _compute_sem(self, t, q):
        term_embedding = self._get_word_embedding(t)
        query_embeddings = [self._get_word_embedding(w) for w in q.split()]
        cosines = [cosine_similarity([term_embedding], [embedding])[0,0] for embedding in query_embeddings]
        return np.max(cosines)

    def _get_word_embedding(self, w):
        return self.model[w] if w in self.model.vocab else np.zeros(self.model.vector_size)


#-----------------------------------------------#
#           EMBEDDING-BASED RETRIEVERS          #
#-----------------------------------------------#
class EmbeddingRetriever:
    def __init__(self, model, vocab, dimension, pooling_strategy, retrieval_corpus):
        self.model = model
        self.vocab = vocab
        self.dimension = dimension
        self.pooling_strategy = pooling_strategy
        self.doc_embeddings = self._embed_corpus(retrieval_corpus, 'articles')

    def search_all(self, queries, top_k, dist_metric):
        query_embeddings = self._embed_corpus(queries, 'queries')
        print("Computing scores...")
        scores = pairwise_distances(X=list(query_embeddings.values()), 
                                    Y=list(self.doc_embeddings.values()),
                                    metric=dist_metric)
        results = np.argsort(scores, axis=1)[:, :top_k] + 1  #+1 because doc ID starts at 1.
        print("Done.")
        return results

    def get_doc_embeddings(self):
        return self.doc_embeddings

    def get_vocab(self):
        return self.vocab

    def _embed_corpus(self, corpus, desc):
        embeddings = dict()
        for i, doc in enumerate(tqdm(corpus, desc='Embedding ' + desc)):
            embeddings[i+1] = self._embed(doc)
        return embeddings

    def _embed(self, text):
        word_embeddings = self._get_word_embeddings(text)
        if not word_embeddings:
            return np.zeros(self.dimension)
        return self._perform_pooling(word_embeddings)

    def _perform_pooling(self, embeddings):
        if self.pooling_strategy == 'mean':
            return np.mean(embeddings, axis=0)
        elif self.pooling_strategy == 'max':
            return np.max(embeddings, axis=0)
        elif self.pooling_strategy == 'sum':
            return np.sum(embeddings, axis=0)
        else:
            sys.exit("Pooling strategy not known, please use 'mean' or 'max'.")

    def _get_word_embeddings(self, text):
        return [self.model[w] for w in text.split()]


class Word2vecRetriever(EmbeddingRetriever):
    def __init__(self, model_path_or_name, pooling_strategy, retrieval_corpus):
        model = KeyedVectors.load_word2vec_format(model_path_or_name, binary=True, unicode_errors="ignore")
        super().__init__(model=model,
                         vocab=set(model.vocab.keys()),
                         dimension=model.vector_size,
                         pooling_strategy=pooling_strategy,
                         retrieval_corpus=retrieval_corpus)

    def _get_word_embeddings(self, text):
        return [self.model[w] for w in text.split() if w in self.vocab]


class FasttextRetriever(EmbeddingRetriever):
    def __init__(self, model_path_or_name, pooling_strategy, retrieval_corpus):
        model = fasttext.load_model(model_path_or_name)
        super().__init__(model=model,
                         vocab=set(model.words),
                         dimension=model.get_dimension(),
                         pooling_strategy=pooling_strategy,
                         retrieval_corpus=retrieval_corpus)


class BERTRetriever(EmbeddingRetriever):
    def __init__(self, model_path_or_name, pooling_strategy, retrieval_corpus):
        self.tokenizer, model = self._load_model(model_path_or_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(self.device)
        super().__init__(model=model,
                         vocab=self.tokenizer.get_vocab(),
                         dimension=model.config.hidden_size,
                         pooling_strategy=pooling_strategy,
                         retrieval_corpus=retrieval_corpus)

    def _load_model(self, model_name):
        if model_name == "camembert":
            tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
            model = CamembertModel.from_pretrained("camembert-base", output_hidden_states=True)
        elif model_name == "flaubert":
            tokenizer = FlaubertTokenizer.from_pretrained("flaubert/flaubert_base_cased")
            model = FlaubertModel.from_pretrained("flaubert/flaubert_base_cased", output_hidden_states=True)
        elif model_name == "mbert":
            tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
            model = BertModel.from_pretrained("bert-base-multilingual-cased", output_hidden_states=True)
        elif model_name == "distilmbert":
            tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            model = DistilBertModel.from_pretrained("distilbert-base-uncased", output_hidden_states=True)
        else:
            sys.exit("Model not known. Exiting...")
        return tokenizer, model


    def _embed(self, text, chunk_size=200, overlap=20, pooling_layer=-2):
        """
        Remarks:
        - Why not using the hidden state of [CLS]? Because a pre-trained model is not fine-tuned on any downstream
        tasks yet, the hidden state of [CLS] is not a good sentence representation.
        - Why not averaging the embeddings from the last hidden layer? The last layer is too closed to the target 
        functions (i.e. masked language model and next sentence prediction) during pre-training, therefore may 
        be biased to those targets.
        - Which hidden layer should be use then? As a rule of thumb, use second-to-last layer. Intuitively, the last 
        hidden layer is close to the training output, so it may be biased to the training targets and lead to a bad         
        representation if you don't fine-tune the model later on. On the other hand, the first hidden layer is close 
        to the word embedding and will probably preserve the very original word information (very little self-attention 
        involved). That said, anything in between ([1, 11]) is then a trade-off.
        - Source: https://bert-as-service.readthedocs.io/en/latest/section/faq.html
        """
        # Convert entire text sequence to token ids. Then, split tokenized sequence into chunks of 'chunk_size' with an 'overlap'.
        # Then make sure that the last chunk is not simply the repeated overlap of the previous chunk.
        # Finally, pad the last chunk to the defined 'chunk_size'. All model inputs will therefore have exactly the same length.
        token_ids = self.tokenizer.encode(text, add_special_tokens=True, truncation=False)
        split_token_ids = [token_ids[i:i+chunk_size] for i in range(0, len(token_ids), chunk_size-overlap)]
        split_token_ids = split_token_ids[:-1] if len(split_token_ids[-1]) <= overlap and len(split_token_ids) > 1 else split_token_ids
        split_token_ids[-1] += [self.tokenizer.pad_token_id] * (chunk_size - len(split_token_ids[-1]))

        # Pass input text to model.
        self.model.eval()
        input_ids = torch.tensor(split_token_ids).to(self.device)
        with torch.no_grad():
            output = self.model(input_ids).hidden_states #tuple of #layers tensors of shape [batch_size x batch_length x D]
            output = torch.stack(output, dim=0) #tensor of shape [#layers x batch_size x batch_length x D]
            output = output.permute(0, 2, 1, 3) #tensor of shape [#layers x batch_length x batch_size x D]

        # Extract token embeddings, then pool to get chunk embeddings, then pool to get final doc embedding.
        token_embeddings = output[pooling_layer] #tensor of shape [batch_length x batch_size x D]
        chunk_embeddings = self._perform_pooling(token_embeddings)
        doc_embedding = self._perform_pooling(chunk_embeddings)
        return doc_embedding.detach().cpu().numpy()


    def _perform_pooling(self, embeddings):
        if self.pooling_strategy == 'mean':
            return torch.mean(embeddings, dim=0)
        elif self.pooling_strategy == 'max':
            return torch.max(embeddings, dim=0).values
        elif self.pooling_strategy == 'sum':
            return torch.sum(embeddings, dim=0)
        else:
            sys.exit("Pooling strategy not known, please use 'mean', 'max' or 'sum'.")

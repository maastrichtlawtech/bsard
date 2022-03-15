import re
from typing import List, Dict, Tuple, Iterable, Type, Union, Optional, Set

import spacy
import pandas as pd
from tqdm import tqdm
tqdm.pandas(desc='Processing text')

from torch.utils.data import Dataset


class TextPreprocessor():
    def __init__(self, spacy_model):
        self.nlp = spacy.load(spacy_model)

    def preprocess(self, series, lowercase=True, remove_punct=True, 
                   remove_num=True, remove_stop=True, lemmatize=True):
        return (series.progress_apply(lambda text: self.preprocess_text(text, lowercase, remove_punct, remove_num, remove_stop, lemmatize)))

    def preprocess_text(self, text, lowercase, remove_punct,
                        remove_num, remove_stop, lemmatize):
        if lowercase:
            text = self._lowercase(text)
        doc = self.nlp(text)
        if remove_punct:
            doc = self._remove_punctuation(doc)
        if remove_num:
            doc = self._remove_numbers(doc)
        if remove_stop:
            doc = self._remove_stop_words(doc)
        if lemmatize:
            text = self._lemmatize(doc)
        else:
            text = self._get_text(doc)
        return text

    def _lowercase(self, text):
        return text.lower()
    
    def _remove_punctuation(self, doc):
        return [t for t in doc if not t.is_punct]
    
    def _remove_numbers(self, doc):
        return [t for t in doc if not (t.is_digit or t.like_num or re.match('.*\d+', t.text))]

    def _remove_stop_words(self, doc):
        return [t for t in doc if not t.is_stop]

    def _lemmatize(self, doc):
        return ' '.join([t.lemma_ for t in doc])

    def _get_text(self, doc):
        return ' '.join([t.text for t in doc])



class BSARDataset(Dataset):
    def __init__(self, queries: pd.DataFrame, documents: pd.DataFrame):
        self.queries = self.get_id_query_pairs(queries) #qid -> query
        self.documents = self.get_id_document_pairs(documents) #docid -> document
        self.one_to_one_pairs = self.get_one_to_one_relevant_pairs(queries) #qid -> rel_docid_i
        self.one_to_many_pairs =  self.get_one_to_many_relevant_pairs(queries) #qid -> {rel_docid_1, ..., rel_docid_n}

    def __len__(self):
        return len(self.one_to_one_pairs)

    def __getitem__(self, idx):
        qid, pos_id = self.one_to_one_pairs[idx]
        return self.queries[qid], self.documents[pos_id]

    def get_id_query_pairs(self, queries: pd.DataFrame) -> Dict[str, str]:
        return queries.set_index('id')['question'].to_dict()

    def get_id_document_pairs(self, documents: pd.DataFrame) -> Dict[str, str]:
        return documents.set_index('id')['article'].to_dict()

    def get_one_to_many_relevant_pairs(self, queries: pd.DataFrame) -> Dict[str, Set[str]]:
        return queries.set_index('id')['article_ids'].str.split(',').apply(lambda x: [int(i) for i in x]).to_dict()

    def get_one_to_one_relevant_pairs(self, queries: pd.DataFrame) -> List[Tuple[int, int]]:
        return (queries
                .assign(article_ids=lambda d: d['article_ids'].str.split(','))
                .set_index(queries.columns.difference(['article_ids']).tolist())['article_ids']
                .apply(pd.Series)
                .stack()
                .reset_index()
                .drop(['category','subcategory','extra_description','question','level_5'], axis=1)
                .rename(columns={0:'article_id','id':'question_id'})
                .apply(pd.to_numeric)
                .sample(frac=1, random_state=42).reset_index(drop=True)
                .to_records(index=False))


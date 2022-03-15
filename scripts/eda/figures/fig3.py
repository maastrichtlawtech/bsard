import os
import itertools
import pandas as pd

import dash
from dash import html
from dash import dcc
import plotly.express as px

import utils
from settings import THEME, COLORS


#------------------------#
# Data                   #
#------------------------#
dfA = pd.read_csv(os.path.abspath(os.path.join(__file__ , "../../../../data/bsard_v1/articles_fr.csv")))
dfQ_train = pd.read_csv(os.path.abspath(os.path.join(__file__ , "../../../../data/bsard_v1/questions_fr_train.csv")))
dfQ_test = pd.read_csv(os.path.abspath(os.path.join(__file__ , "../../../../data/bsard_v1/questions_fr_test.csv")))
dfQ = pd.concat([dfQ_train, dfQ_test])

#------------------------#
# Plotly figure          #
#------------------------#
def make_figure(dfArt, dfQuest):
    return (dfArt
            .pipe(utils.start_pipeline)
            .pipe(add_isref_feature, set(itertools.chain.from_iterable(dfQuest['article_ids'].str.split(',').apply(lambda x: list(map(int, x))))))
            .pipe(count_articles_by_code)
            .pipe(utils.translate_to_en, 'code', utils.EN_CODES)
            .pipe(plot_articles_as_barchart))


def add_isref_feature(dataf, reference_ids):
    """
    Args:
        dataf (pandas.DataFrame): the 'articles' dataframe.
    Returns:
        pandas.DataFrame: the input dataframe with a new 'isref' column indicating if the article
        appears as a legal reference for one or several questions in the Question-Reference dataset.
    """
    dataf['isref'] = dataf['id'].apply(lambda x: 'Yes' if x in reference_ids else 'No')
    return dataf

def count_articles_by_code(dataf):
    """
    Args:
        dataf (pandas.DataFrame): the 'articles' dataframe with the 'isref' column.
    Returns:
        pandas.DataFrame: a new 3-column dataframe of the form {'code','isref','count'} that lists the
        number of articles in each code while differentiating those that are legal references in the
        Question-Reference dataset from those who aren't.
    """
    return (dataf
            .groupby(by=["code", "isref"], as_index=False)
            .agg(count=("id", "count"))
            .sort_values('count', ascending=False))

def plot_articles_as_barchart(dataf):
    """
    Args:
        dataf (pandas.DataFrame): a 3-column dataframe of the form {'code','isref','count'}.
    Returns:
        plotly.graph_objects.Figure: a Bar Chart that shows the number of articles by code while differentiating 
        between those that are legal references in the Question-Reference dataset from those who aren't.
    """
    return (px.bar(dataf,
                    y="code", x="count", color="isref",
                    labels={'code':'Code', 'count':'Articles', 'isref': 'Is a reference'},
                    width=1500, height=770,
                    template=COLORS[THEME]['plotly'])
                .update_traces(texttemplate='%{x}', textposition='inside'))

#------------------------#
# Dash layout            #
#------------------------#
layout = html.Div([
    html.H4(children='How many articles are there in the retrieval corpus?', style={'color': COLORS[THEME]['dash']['text']}),
    dcc.Graph(figure=make_figure(dfA, dfQ)),
])

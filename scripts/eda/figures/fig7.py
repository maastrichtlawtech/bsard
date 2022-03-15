import os
import pandas as pd

import dash
from dash import html
from dash import dcc
import plotly.express as px

from settings import THEME, COLORS
from utils import (start_pipeline, 
                   create_one_row_per_ref, 
                   add_nitems_feature, 
                   merge_with_articles, 
                   break_text)


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
def make_figure(dfQuest, dfArt):
    return (dfQuest
            .pipe(start_pipeline)
            .pipe(create_one_row_per_ref)
            .pipe(group_questions_by_reference)
            .pipe(add_nitems_feature, 'question_id')
            .pipe(merge_with_articles, dfArt)
            .pipe(plot_questions_per_ref_barplot))

def group_questions_by_reference(dataf):
    """
    Args:
        dataf (pandas.DataFrame): a transformed version of the 'legal_question_reference_fr' dataframe where 
            each row is a match between one question and one reference.
    Returns:
        pandas.DataFrame: a two-column dataframe of the form {'article_id','question_ids'}, where each unique
        reference relates to its parent question(s).
    """
    return (dataf
            .groupby("article_id", as_index=False)
            .agg({'question_id': lambda s: ','.join(map(str, s))}))

def plot_questions_per_ref_barplot(dataf):
    """
    Args:
        dataf (pandas.DataFrame): the 'legal_question_reference_fr' dataframe.
    Returns:
        plotly.graph_objects.Figure: a Bar Chart that shows the number of questions per reference.
    """
    dataf['count'] = 1
    dataf['question_id'] = break_text(dataf['question_id'], width=30)
    dataf['description'] = break_text(dataf['description'], width=70)
    return (px.bar(dataf,
                    x="nitems", y="count", color="code",
                    labels={'nitems':'Number of mentions', 'count':'References count', 'code':'Code'},
                    custom_data=['article_id', 'description', 'question_id'],
                    color_discrete_sequence=px.colors.qualitative.Light24,
                    width=1500, height=600,
                    template=COLORS[THEME]['plotly'])
                .update_traces(hovertemplate="<br>".join([#"ColX: %{x}", "ColY: %{y}",
                                                          "<b>Article ID</b>: %{customdata[0]}",
                                                          "<b>Description</b>: %{customdata[1]}",
                                                          "<b>Question ID(s)</b>: %{customdata[2]}"]))
                .update_xaxes(nticks=20, ticks="outside"))

#------------------------#
# Dash layout            #
#------------------------#
layout = html.Div([
    html.H4(children='How often are references mentioned?', style={'color': COLORS[THEME]['dash']['text']}),
    dcc.Graph(figure=make_figure(dfQ, dfA)),
])

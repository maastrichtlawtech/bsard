import os
import pandas as pd

import dash
from dash import html
from dash import dcc
import plotly.express as px

from settings import THEME, COLORS
from utils import start_pipeline, create_subplots_grid, plot_violinbox


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
    return create_subplots_grid(figures=[dfArt
                                            .pipe(start_pipeline)
                                            .pipe(add_length_feature, 'article')
                                            .pipe(plot_violinbox, 'length', 'Number of words'),
                                         (dfQuest
                                            .pipe(start_pipeline)
                                            .pipe(add_length_feature, 'question')
                                            .pipe(plot_violinbox, 'length', 'Number of words'))],
                                ncols=2, col_widths=[0.5,0.5],
                                titles=['Articles','Questions'])

def add_length_feature(dataf, column_name):
    """
    Args:
        dataf (pandas.DataFrame): either the 'questions' or the 'articles' dataframe.
        column_name (str): the name of the column containing the text chunks.
    Returns:
        pandas.DataFrame: the input dataframe with a new 'length' column indicating the length of each text chunk from
        the '<column_name>' column.
    """
    return (dataf
            .assign(length=lambda d: d[column_name].str.len()))


#------------------------#
# Dash layout            #
#------------------------#
layout = html.Div([
    html.H4(children='How long are the questions and articles?', style={'color': COLORS[THEME]['dash']['text']}),
    dcc.Graph(figure=make_figure(dfA, dfQ)),
])

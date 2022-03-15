import os
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
def make_figure(dfQuest, dfArt):
    return (dfQuest
            .pipe(utils.start_pipeline)
            .pipe(utils.translate_to_en, 'category', utils.EN_CATEGORY)
            .pipe(utils.create_one_row_per_ref)
            .pipe(utils.merge_with_articles, dfArt)
            .pipe(utils.translate_to_en, 'code', utils.EN_CODES)
            .pipe(plot_parallel_categories, color_by='category'))


def plot_parallel_categories(dataf, color_by):
    """
    Args:
        dataf (pandas.DataFrame): a merged dataframe of the 'questions' and  'articles' dataframes
                    on the 'article_id' column.
        color_by (str): the name of the column to color on.
    Returns:
        plotly.graph_objects.Figure: a Parallel Category Chart relating the question categories to the refered codes.
    """
    return (px.parallel_categories(dataf, 
                                    dimensions=['category', 'code'],
                                    labels={'category':'Question type', 'code':'Refered code'},
                                    color=dataf.assign(color_id=lambda d: d[color_by].astype('category').cat.codes)['color_id'],
                                    color_continuous_scale=px.colors.qualitative.Plotly,
                                    width=1600, height=770,
                                    template=COLORS[THEME]['plotly'])
                .update_layout(coloraxis_showscale=False, margin=dict(l=70, r=250, t=40, b=40)))

#------------------------#
# Dash layout            #
#------------------------#
layout = html.Div([
    html.H4(children='What code do the questions refer to?', style={'color': COLORS[THEME]['dash']['text']}),
    dcc.Graph(figure=make_figure(dfQ, dfA)),
])

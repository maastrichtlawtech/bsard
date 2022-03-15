import os
import pandas as pd

import dash
from dash import dcc
from dash import html
import plotly.express as px

import utils
import settings


#------------------------#
# Data                   #
#------------------------#
dfQ_train = pd.read_csv(os.path.abspath(os.path.join(__file__ , "../../../../data/bsard_v1/questions_fr_train.csv")))
dfQ_test = pd.read_csv(os.path.abspath(os.path.join(__file__ , "../../../../data/bsard_v1/questions_fr_test.csv")))
dfQ = pd.concat([dfQ_train, dfQ_test])

#------------------------#
# Plotly figure          #
#------------------------#
def make_figure(dataf_train, dataf_test):
    return utils.create_subplots_grid(figures=[dataf_train
                                                .pipe(utils.start_pipeline)
                                                .pipe(utils.translate_to_en, 'category', utils.EN_CATEGORY)
                                                .pipe(utils.translate_to_en, 'subcategory', utils.EN_SUBCATEGORY)
                                                .pipe(plot_categories_as_sunburst, "Training questions"),
                                            dataf_test
                                                .pipe(utils.start_pipeline)
                                                .pipe(utils.translate_to_en, 'category', utils.EN_CATEGORY)
                                                .pipe(utils.translate_to_en, 'subcategory', utils.EN_SUBCATEGORY)
                                                .pipe(plot_categories_as_sunburst, "Testing questions")],
                                ncols=2, col_widths=[0.5, 0.5])

def plot_categories_as_sunburst(dataf, title):
    """
    Args:
        dataf (pandas.DataFrame): the 'questions' dataframe.
    Returns:
        plotly.graph_objects.Figure: a Sunburst Chart of the questions' category and subcategory.
    """
    dataf['tmp'] = title
    dataf = dataf.fillna('-')
    return (px.sunburst(dataf,
                        path=['tmp', 'category', 'subcategory'],
                        labels={'count':'Questions'},
                        width=1200, height=800,
                        template=utils.COLORS[utils.THEME]['plotly'])
              .update_traces(textinfo="label+percent root", insidetextorientation='radial',
                             hovertemplate="<br>".join(["<b>%{label}</b>",
                                                        "Questions: %{value}",
                                                        "Root ratio: %{percentRoot:%.3Òf}",
                                                        "Parent ratio: %{percentParent:%.3f}"]))
             .update_layout(uniformtext=dict(minsize=30, mode=False)))

def plot_categories_as_treemap(dataf):
    """
    Args:
        dataf (pandas.DataFrame): the 'questions' dataframe.
    Returns:
        plotly.graph_objects.Figure: a Treemap Chart of the questions' category and subcategory.
    """
    dataf = dataf.fillna('-')
    return (px.treemap(dataf,
                        path=['category', 'subcategory'],# 'topic'],
                        labels={'count':'Questions'},
                        width=1600, height=800,
                        template=utils.COLORS[utils.THEME]['plotly'])
              .update_traces(textinfo="label+percent root",
                             hovertemplate="<br>".join(["<b>%{label}</b>",
                                                        "Questions: %{value}",
                                                        "Root ratio: %{percentRoot:%.2f}",
                                                        "Parent ratio: %{percentParent:%.2f}"])))

#------------------------#
# Dash layout            #
#------------------------#
layout = html.Div([
    html.H4(children='What are the questions about?', style={'color': utils.COLORS[utils.THEME]['dash']['text']}),
    dcc.Graph(figure=plot_categories_as_treemap(
                        dfQ.pipe(utils.start_pipeline)
                            .pipe(utils.translate_to_en, 'category', utils.EN_CATEGORY)
                            .pipe(utils.translate_to_en, 'subcategory', utils.EN_SUBCATEGORY))),
    dcc.Graph(figure=make_figure(dfQ_train, dfQ_test)),
])

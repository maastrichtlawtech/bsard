import os
import pandas as pd

import dash
from dash import html
from dash import dcc
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
def make_figure(dataf):
    return utils.create_subplots_grid(figures=[
                                        dataf
                                            .pipe(utils.start_pipeline)
                                            .pipe(utils.add_nitems_feature, 'article_ids')
                                            .pipe(plot_nrefs_barplot),
                                        dataf
                                            .pipe(utils.start_pipeline)
                                            .pipe(utils.add_nitems_feature, 'article_ids')
                                            .pipe(plot_nrefs_piechart, maxdisplay=15)],
                                ncols=2, col_widths=[0.7,0.3])

def plot_nrefs_barplot(dataf):
    """
    Args:
        dataf (pandas.DataFrame): the 'questions' dataframe.
    Returns:
        plotly.graph_objects.Figure: a Bar Chart that shows the number of references per question.
    """
    dataf['count'] = 1
    dataf['question'] = utils.break_text(dataf['question'], width=50)
    dataf['article_ids'] = utils.break_text(dataf['article_ids'], width=50)
    return (px.bar(dataf,
                    x="nitems", y="count",
                    labels={'nitems':'Reference(s)/question', 'count':'Questions count'},
                    custom_data=['id','question','article_ids'],
                    width=1500, height=600,
                    template=utils.COLORS[utils.THEME]['plotly'])
                .update_traces(opacity=0.8,
                               hovertemplate="<br>".join([#"ColX: %{x}", "ColY: %{y}",
                                                          "<b>Question ID</b>: %{customdata[0]}",
                                                          "<b>Question</b>: %{customdata[1]}",
                                                          "<b>Reference ID(s)</b>: %{customdata[2]}"]),
                               marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)', marker_line_width=1.5)
                .update_xaxes(nticks=20, ticks="outside"))


def plot_nrefs_piechart(dataf, maxdisplay=10):
    """
    Args:
        dataf (pandas.DataFrame): the 'questions' dataframe.
    Returns:
        plotly.graph_objects.Figure: a Pie Chart that shows the number of references per question.
    """
    dataf = (dataf
                .groupby("nitems", as_index=False)
                .agg(count=("id", "count")))
    dataf = pd.concat([
                dataf[dataf['nitems'] <= maxdisplay], 
                pd.DataFrame({'nitems':[str(maxdisplay)+'+'], 'count':[dataf.loc[dataf['nitems'] > maxdisplay, "count"].sum()]})])
    return (px.pie(dataf, 
                    values='count', names='nitems', 
                    labels={'nitems':'Reference(s)/question', 'count':'Questions'},
                    width=1500, height=600,
                    template=utils.COLORS[utils.THEME]['plotly']))


#------------------------#
# Dash layout            #
#------------------------#
layout = html.Div([
    html.H4(children='How many references do questions have?', style={'color': utils.COLORS[utils.THEME]['dash']['text']}),
    dcc.Graph(figure=make_figure(dfQ)),
])

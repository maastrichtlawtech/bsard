import os
import numpy as np
import pandas as pd

import dash
from dash import html
from dash import dcc
import plotly.express as px

import base64
from io import BytesIO
import matplotlib.pyplot as plt
from wordcloud import WordCloud

import utils
from settings import THEME, COLORS


#------------------------#
# Data                   #
#------------------------#
dfA = pd.read_csv(os.path.abspath(os.path.join(__file__ , "../../../../data/bsard_v1/articles_fr.csv")))

#------------------------#
# Plotly figure          #
#------------------------#
def make_figure(dataf):
    img = (dfA
            .pipe(utils.start_pipeline)
            .pipe(plot_wordclouds, text_col='article', split_col='code', ncols=2))
    return 'data:image/png;base64,{}'.format(base64.b64encode(img.getvalue()).decode())


def create_wordcloud(texts):
    """
    Args:
        texts (list[str]): a list of text chunks.
    Returns:
        wordcloud.WordCloud: a Wordcloud Plot showing the most frequent words from the list of text chunks.
    """
    return (WordCloud(width=900, height=750,
                      random_state=42, collocations=False, 
                      background_color=COLORS[THEME]['dash']['background'],
                      colormap='Set1', 
                      stopwords=utils.STOPWORDS)
                .generate(' '.join(texts)))


def plot_wordclouds(dataf, text_col, split_col, ncols=2):
    """
    Args:
        dataf (pandas.DataFrame): the 'articles_fr' dataframe.
        text_col (str): name of the column in dataf that contains the texts.
        split_col (str): name of the column in dataf on which to plot various wordclous.
        ncols (int, optional): number of columns in figure. Default to 2.
    Returns:
        matplotlib.pyplot.figure: a nrows x ncols figure of Wordcloud Plots for each '<split_col>'.
    """
    split_on_values = dataf[split_col].unique()
    nrows=int(np.ceil(len(split_on_values)/ncols))

    fig, axarr = plt.subplots(ncols=ncols, nrows=nrows, figsize=(16, 7.5*nrows))
    fig.patch.set_facecolor(COLORS[THEME]['dash']['background'])

    for i, code in enumerate(split_on_values):
        texts = dataf[dataf.code == code][text_col].values
        cloud = create_wordcloud(texts)
        
        idx = i+1 if i%ncols==0 else i
        axarr[int(np.ceil(idx/ncols))-1, i%ncols].imshow(cloud)
        axarr[int(np.ceil(idx/ncols))-1, i%ncols].set_title(code, fontdict={'color':COLORS[THEME]['dash']['text'], 'size':10})
        axarr[int(np.ceil(idx/ncols))-1, i%ncols].axis('off')

    img = BytesIO()
    plt.savefig(img, format="PNG")
    return img

#------------------------#
# Dash layout            #
#------------------------#
layout = html.Div([
    html.H4(children='What are the most frequent words in the articles?', style={'color': COLORS[THEME]['dash']['text']}),
    html.Img(src=make_figure(dfA)),
])

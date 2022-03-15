# -*- coding: utf-8 -*-
import os
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.express as px

from settings import THEME, COLORS, TABS_STYLE, TAB_STYLE, SELECTED_TAB_STYLE
from figures import fig1, fig2, fig3, fig4, fig5, fig6, fig7


# Initialise the app.
app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
app.title = 'The Belgian Statutory Article Retrieval Dataset (BSARD)'

# Define the app.
app.layout = html.Div(style={'backgroundColor': COLORS[THEME]['dash']['background'], 'margin': -8}, children=[
    html.H2(children='Belgian Statutory Article Retrieval Dataset (BSARD)', style={'color': COLORS[THEME]['dash']['text']}),
    dcc.Tabs(id='tabs', value='tab-1',
            children=[dcc.Tab(label=str(i+1), 
                                value='tab-'+str(i+1), 
                                style=TAB_STYLE,
                                selected_style=SELECTED_TAB_STYLE)
                        for i, _ in enumerate(next(os.walk(os.path.abspath(os.path.join(__file__ , "../figures"))))[2])],
            style=TABS_STYLE),
    html.Div(id='tabs-content')
])

@app.callback(Output('tabs-content', 'children'), [Input('tabs', 'value')])
def render_content(tab):
    if tab == 'tab-1':
        return fig1.layout
    elif tab == 'tab-2':
        return fig2.layout
    elif tab == 'tab-3':
        return fig3.layout
    elif tab == 'tab-4':
        return fig4.layout
    elif tab == 'tab-5':
        return fig5.layout
    elif tab == 'tab-6':
        return fig6.layout
    elif tab == 'tab-7':
        return fig7.layout


if __name__ == '__main__':
    app.run_server(debug=False)

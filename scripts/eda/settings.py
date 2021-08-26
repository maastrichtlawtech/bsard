# -*- coding: utf-8 -*-

THEME = 'white'

COLORS = {
    'white':
            {'dash':{'background': '#FFFFFF', 'text': '#111111'},
            'plotly':'plotly_white'},
    'black':
            {'dash':{'background': '#111111', 'text': '#FFFFFF'},
            'plotly':'plotly_dark'},
}

TABS_STYLE = {
    'height': '44px',
    'padding': '0px 30px',
    'borderBottom': '1px solid lightgrey',
}

TAB_STYLE = {
    'color': COLORS[THEME]['dash']['text'],
    'background-color': COLORS[THEME]['dash']['background'],
    'borderTop': '3px solid transparent',
    'borderLeft': '0px',
    'borderRight': '0px',
    'borderBottom': '0px',
}

SELECTED_TAB_STYLE = {
    'fontWeight': 'bold',
    'color': COLORS[THEME]['dash']['text'],
    'background-color': COLORS[THEME]['dash']['background'],
    'box-shadow': '1px 1px 0px'+COLORS[THEME]['dash']['background'],
    'borderLeft': '1px solid lightgrey',
    'borderRight': '1px solid lightgrey',
    'borderTop': '4px solid #a1cae2',
    'border-top-left-radius': '5px',
    'border-top-right-radius': '5px',
}
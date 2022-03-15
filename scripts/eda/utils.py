import datetime as dt
import textwrap

import pandas as pd
import numpy as np
import spacy
import nltk
#nltk.download('stopwords')

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from settings import THEME, COLORS


STOPWORDS = set(nltk.corpus.stopwords.words('french')) | spacy.load("fr_core_news_md").Defaults.stop_words

EN_CATEGORY = {
    'Famille': 'Family', 
    'Logement': 'Accomodation', 
    'Argent': 'Money', 
    'Justice': 'Justice', 
    'Etrangers':'Foreigners', 
    'Protection sociale':'Social security', 
    'Travail': 'Work'
}
EN_SUBCATEGORY = {
    'Aide pour gérer le budget':'Help with budget management',
    'Assurances':'Insurance',
    'Dettes':'Debts', 
    'Fiscalité':'Taxes',
    'Impôts':'Taxes', 
    'Protection du consommateur':'Consumer protection', 
    'Succession et donation':'Inheritance and donation',
    'Etrangers et famille':'Foreigners and family', 
    'Etrangers et travail salarié':'Foreigners and paid employment',
    'Mineurs étrangers':'Foreign minors', 
    'Nationalité belge':'Belgian citizenship',  
    'Séjour illégal':'Illegal stay',
    'Séjour non-européen':'Non-European stay', 
    'Lien parents/enfants':'Parent/child relationship',
    'Mineurs en danger':'Minors in danger',
    'Obligations alimentaires':'Maintenance obligations',
    "Personnes à l'autonomie fragilisée":'Persons with weakened autonomy',
    'Régler un conflit familial':'Settling a family conflict',
    'Situation de couples':"Couples' situation",
    'Vivre en couple':'Living as a couple',
    'Être parents':'Being a parent',
    'Au tribunal':'In court',
    'Casier judiciaire':'Criminal record',
    'Infractions':'Offenses',
    "L'avocat":'The lawyer',
    'Petite délinquance':'Petty crime', 
    'Bail de résidence principale (Wallonie)':'Lease of main residence (Wallonia)',
    'Domicile':'Domicile', 
    'Hébergement':'Accommodation',
    'Insalubrité (Wallonie)':'Insalubrity (Wallonia)',
    'Insalubrité en Wallonie':'Insalubrity (Wallonia)',
    'Location en Wallonie':'Rental in Wallonia',
    'Location à Bruxelles':'Rental in Brussels',
    'Régler un conflit locatif': 'Settling a rental dispute',
    'Urbanisme':'Urbanism',
    'Voisinage':'Neighborhood',
    'Aide du CPAS':'Help from the CPAS',
    'Chômage':'Unemployment',
    'Contester une décision de sécurité sociale':'Contesting a social security decision',
    'Grossesse et naissance':'Pregnancy and birth',
    'Santé et maladie':'Health and illness',
    'Vieillesse - pensions':'Old age - pensions',
    "Entretien d'embauche":'Job interview',
    'Les risques du travail non déclaré': 'The risks of undeclared work',
    'Maladie - incapacité de travail':'Illness - incapacity for work',
    'Maladie et accident':'Illness and accident',
    'Secret professionnel':'Professional secrecy',
    'Travail et parentalité':'Work and parenthood',
    'Travail salarié':'Salaried work'
}

EN_CODES = {
    'Code Civil':'Civil Code',
    "Code d'Instruction Criminelle":'Code of Criminal Instruction',
    'Code Judiciaire':'Judicial Code',
    'La Constitution':'The Constitution',
    'Code de la Nationalité Belge':'Code of Belgian Nationality',
    'Code Pénal':'Penal Code',
    'Code Pénal Social':'Social Penal Code',
    'Code Pénal Militaire':'Military Penal Code',
    'Code de la Démocratie Locale et de la Décentralisation':'Code of Local Democracy and Decentralization',
    'Code de Droit Economique':'Code of Economic Law',
    'Codes des Droits et Taxes Divers':'Code of Various Rights and Taxes',
    'Code de Droit International Privé':'Code of Private International Law',
    'Code des Sociétés et des Associations':'Code of Companies and Associations',
    'Code du Bien-être au Travail':'Code of Workplace Welfare',
    'Code Electoral':'Electoral Code',
    'Code Consulaire':'Consular Code', 
    'Code Ferroviaire':'Railway Code',
    'Code de la Navigation':'Navigation Code',
    'Code Forestier':'Forestry Code',
    'Code Rural':'Rural Code',
    'Code de la Fonction Publique Wallonne':'Walloon Public Service Code',
    "Code Wallon de l'Enseignement Fondamental et de l'Enseignement Secondaire":'Walloon Code of Basic and Secondary Education',
    "Code Wallon de l'Agriculture":"Walloon Code of Agriculture",
    "Code Wallon de l'Habitation Durable":"Walloon Code of Sustainable Housing",
    'Code Wallon du Bien-être des animaux':"Walloon Animal Welfare Code",
    'Code Wallon du Développement Territorial':'Walloon Code of Territorial Development',
    "Code Wallon de l'Action sociale et de la Santé":"Walloon Code of Social Action and Health",
    "Code Réglementaire Wallon de l'Action sociale et de la Santé":"Walloon Regulatory Code of Social Action and Health",
    "Code Wallon de l'Environnement":"Walloon Code of the Environment",
    "Code de l'Eau intégré au Code Wallon de l'Environnement":"Water Code integrated into the Walloon Environment Code",
    "Code Bruxellois de l'Aménagement du Territoire":"Brussels Spatial Planning Code",
    "Code Bruxellois de l'Air, du Climat et de la Maîtrise de l'Energie":'Brussels Code of Air, Climate and Energy Management',
    'Code Bruxellois du Logement':'Brussels Housing Code',
    'Code Electoral Communal Bruxellois':'Brussels Municipal Electoral Code'
}


def log_step(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        tic = dt.datetime.now()
        result = func(*args, **kwargs)
        time_taken = str(dt.datetime.now() - tic)
        print(f"Just ran '{func.__name__}'. Output df shape: {result.shape}. Took: {time_taken}")
        return result
    return wrapper


def start_pipeline(dataf):
    return dataf.copy()


def translate_to_en(dataf, column, translations):
    dataf[column] = dataf[column].map(translations)
    return dataf


def break_text(datas, width):
    """
    Args:
        datas (pandas.Series): a series containing texts to break (split with newline character).
        width (int): length of each newline split.
    Returns:
        pandas.Series: the input series where the texts have been split with newline character.
    """
    return datas.apply(lambda x: "<br>".join(textwrap.wrap(str(x), width=width)))


def create_subplots_grid(figures, titles=None, ncols=2, col_widths=[0.5,0.5]):
    """
    Args:
        figures (list[plotly.graph_objects.Figure]): a list of plotly figures.
        titles (list[str], optional): list of subplot names. Default to None.
        ncols (int, optional): number of columns in the figure. Default to 2.
        col_widths (list[float], optional): widths of each column. Default to [0.5,0.5].
    Returns:
        plotly.graph_objects.Figure: a signle figure combining the given ones.
    """
    # Make sure col_widths values are correct.
    assert len(col_widths) == ncols, 'Number of column widths must be equal to number of columns'
    assert sum(col_widths) == 1.0, 'Sum of column widths must be equal to 1.0'

    # Initialize figure with subplots.
    nrows=int(np.ceil(len(figures)/ncols))
    fig = make_subplots(cols=ncols, rows=nrows,
                        column_widths=col_widths, row_heights=None,
                        shared_xaxes=False, shared_yaxes=False,
                        vertical_spacing=0.1, horizontal_spacing=0.2,
                        specs=np.array([{'type': sub['data'][0]['type']} for sub in figures] 
                                        + ([None]* ((ncols*nrows)-len(figures))))
                                .reshape((nrows,ncols))
                                .tolist(),
                        subplot_titles=titles)
    
    # Add traces.
    for i, subfig in enumerate(figures):
        idx = i+1 if i%ncols==0 else i
        fig.add_trace(subfig['data'][0], row=int(np.ceil(idx/ncols)), col=i%ncols+1)
        fig.update_xaxes(title_text=subfig['layout']['xaxis']['title']['text'], row=int(np.ceil(idx/ncols)), col=i%ncols+1)
        fig.update_yaxes(title_text=subfig['layout']['yaxis']['title']['text'], row=int(np.ceil(idx/ncols)), col=i%ncols+1)
    
    # Update general layout.
    fig.update_layout(width=1600, height=770*nrows, title_text="", template=COLORS[THEME]['plotly'])
    return fig


def create_one_row_per_ref(dataf):
    """
    Args:
        dataf (pandas.DataFrame): the 'legal_question_reference_fr' dataframe.
    Returns:
        pandas.DataFrame: a transformed version of the input dataframe where each row is a match between
        one question and one reference (hence, multiple rows now concern the same question).
    """
    return (dataf
                .assign(article_ids=lambda d: d['article_ids'].str.split(','))
                .set_index(dataf.columns.difference(['article_ids']).tolist())['article_ids']
                .apply(pd.Series)
                .stack()
                .reset_index()
                .drop(['level_5'], axis=1)
                .rename(columns={0:'article_id','id':'question_id'}))


def merge_with_articles(dataf, dfArt):
    """
    Args:
        dataf (pandas.DataFrame): a dataframe containing an 'article_id' column.
        dfArt (pandas.DataFrame): the 'belgian_law_articles_fr' dataframe.
    Returns:
        pandas.DataFrame: the input dataframe with an additional 'reference' column indicating the precise location
            in the law of the corresponding article.
    """
    return (pd.merge(dataf.astype({'article_id': int}),
                    dfArt.rename(columns={'id':'article_id'}), 
                    on="article_id", how="left"))


def add_nitems_feature(dataf, column):
    """
    Args:
        dataf (pandas.DataFrame): a dataframe with a column containing comma separated strings.
        column (str): the name of a column in dataf whose values are comma separated strings.
    Returns:
        pandas.DataFrame: the input dataframe with a new 'nitems' column indicating the number
        of items in each comma separated string from '<column>.
    """
    return dataf.assign(nitems=lambda d: d[column].str.split(',').str.len())


def plot_violinbox(dataf, column_name, ylabel):
    fig = (px.violin(dataf,
                        y=column_name,
                        box=True, points="all", #hover_data=['id'],
                        labels={column_name:ylabel},#,'id': 'ID'},
                        color_discrete_sequence=px.colors.qualitative.G10,
                        width=1500, height=800,
                        template=COLORS[THEME]['plotly'])
            #.update_traces(meanline_visible=True)
        )

    for y in zip(["Median","Max"], dataf[column_name].quantile([0.5,1])):
        fig.add_annotation(y=y[1], x=0.5, text=y[0] + ":" + str(int(y[1])), showarrow=False, font=dict(size=16))
    return fig

import pandas as pd
import numpy as np


def str_to_float(x):
    try:
        x = str(x)
        # x = x.replace(u'\xa0', '')
        x = x.replace(' ', '')
        x = x.replace(',', '.')
        return np.float32(x)
    except:
        return np.float32(0)


def load_df(path):
    df = pd.read_csv(path)
    df = df.transpose().reset_index()
    df.columns = df.iloc[0]
    df = df[1:].rename(columns = {'Артикул': 'Дата'})
    df['Дата'] = pd.to_datetime(df['Дата'], format='%d.%m.%Y')
    df.index = pd.DatetimeIndex(df['Дата'])
    df = df.drop(columns=['Дата'])
    articles = df.columns.tolist()
    bad_art = []
    for art in articles:
        try:
            df[art] = df[art].astype('half')
        except:
            bad_art.append(art)
            
    for art in bad_art:
        df[art] = df[art].apply(str_to_float)

    for art in articles:
        df[art] = df[art].fillna(0)
    return df
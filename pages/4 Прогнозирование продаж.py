import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import custom_module as gap
import os
from statsmodels.tsa import seasonal, stattools
from statsmodels.graphics import tsaplots
from dateutil.relativedelta import relativedelta
from gap_models import *


@st.cache_data
def load_data():
    df = gap.load_df(file)
    return df


st.set_page_config(page_title="Прогноз продаж",
                   layout = 'wide')

absolute_path = os.path.abspath(os.path.dirname('main_window.py'))
file = absolute_path + "/data/sales_plus.csv"

df = load_data()

articles = df.columns.tolist()

trend_models = [
    'Линейный тренд',
    'Логарифмический тренд',
    'Гиперболический тренд',
    'Аддитивная модель',
]

naive_models = [
    'Среднее',
    'Среднее(оконное)',
    'Медиана',
    'Медиана(оконная)',
    'Предыдущим',
    'Сезонная предыдущая'
]

ARIMA_models = [
    'Скользящее среднее(MA)',
    'ARIMA',
    'SARIMAX'
]

ES_models = [
    'Простое экспоненциальное сглаживание(SES)',
    'Экспоненциальное сглаживание(ES)',
    'Модель Хольта',
    'Экспоненциальное сглаживание с трендом(ETS)'
]


art = st.selectbox('Список артикулов', articles)


models = st.multiselect('Наивные модели', naive_models)
models += st.multiselect('Трендовые модели', trend_models)
models += st.multiselect('Модели ARIMA', ARIMA_models)
models += st.multiselect('Модели экспоненциального сглаживания', ES_models)


new_dates = [df.index.max() + relativedelta(months = 1),
             df.index.max() + relativedelta(months = 2),
             df.index.max() + relativedelta(months = 3)]


data = pd.DataFrame(df[art].values.tolist() + [np.nan] * 3,
                    index = df.index.tolist() + new_dates,
                    columns = [art])

data = data.sort_index()


for model in models:
    match model:
        case 'Линейный тренд':
            data[model] = linear_trend(data)
        case 'Логарифмический тренд':
            data[model] = log_trend(data)
        case 'Гиперболический тренд':
            data[model] = hyperbolic_trend(data)
        case 'Аддитивная модель':
            data[model] = additive_ts(data)
        case 'Среднее':
            data[model] = mean_prediction(data)
        case 'Среднее(оконное)':
            data[model] = mean_seasonal_prediction(data)
        case 'Медиана':
            data[model] = median_prediction(data)
        case 'Медиана(оконная)':
            data[model] = median_seasonal_prediction(data)
        case 'Предыдущим':
            data[model] = lag_prediction(data)
        case 'Сезонная предыдущая':
            data[model] = lag_seasonal_prediction(data)
        case 'Скользящее среднее(MA)':
            data[model] = auto_MA(data)
        case 'ARIMA':
            data[model] = auto_ARIMA(data)
        case 'SARIMAX':
            data[model] = auto_SARIMAX(data)
        case 'Простое экспоненциальное сглаживание(SES)':
            data[model] = SES_pred(data)
        case 'Экспоненциальное сглаживание(ES)':
            data[model] = ES_pred(data)
        case 'Модель Хольта':
            data[model] = Holt_pred(data)
        case 'Экспоненциальное сглаживание с трендом(ETS)':
            data[model] = ETS_pred(data)


st.line_chart(data)
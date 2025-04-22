import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import custom_module as gap
import os
from statsmodels.tsa import seasonal, stattools
from statsmodels.graphics import tsaplots


@st.cache_data
def load_data():
    df = gap.load_df(file)
    return df


def get_hist_plot(data):
    fig, ax = plt.subplots()
    ax.hist(data)
    ax.legend(data.name, fontsize = 16)
    plt.title('Гистограмма количества продаж', fontsize = 16)
    return fig


def decompose(ts):
    decomposed = seasonal.seasonal_decompose(ts)

    fig, ax = plt.subplots()
    
    fig.set_size_inches((12, 12))
    plt.subplot(311)
    plt.title('Тренд')
    decomposed.trend.plot(style = 'o-', ax=plt.gca())
    
    plt.subplot(312)
    plt.title('Сезонная компонента')
    decomposed.seasonal.plot(style = 'o-', ax=plt.gca())
    
    plt.subplot(313)
    plt.title('Остатки')
    decomposed.resid.plot(style = 'o-', ax=plt.gca())
    
    fig.subplots_adjust(hspace=.5)
    return fig


def plot_acf(timeseries):
    fig, ax = plt.subplots()
    fig.set_size_inches((15, 8))
    tsaplots.plot_acf(timeseries, lags = len(timeseries)-2, ax = ax)
    plt.title('График автокорреляции' , size = 16)
    plt.xlabel('Лаг', size = 14)
    plt.ylim(-1,1.05)
    return fig


def plot_pacf(timeseries):
    fig, ax = plt.subplots()
    fig.set_size_inches((15, 8))
    tsaplots.plot_pacf(timeseries, lags = int(len(timeseries)-2)/2, ax = ax)
    plt.title('График частичной автокорреляции' , size = 16)
    plt.xlabel('Лаг', size = 14)
    plt.ylim(-1,1.05)
    return fig


st.set_page_config(page_title="История продаж",
                   layout = 'wide')

plt.style.use('seaborn-v0_8')

absolute_path = os.path.abspath(os.path.dirname('main_window.py'))
file = absolute_path + "/data/sales_plus.csv"

df = load_data()

articles = df.columns.tolist()
default_value = articles[0]
art = st.selectbox('Список артикулов', articles)

tab_df, tab_plot, tab_decomp, tab_acf, tab_pacf, tab_adf = (
    st.tabs(['Данные',
             'История продаж',
             'Декомпозиция ряда',
             'График ACF',
             'График PACF',
             'Тест Дики-Фуллера'
            ]))

data = pd.DataFrame(df[art].values, index = df.index, columns = [art])

with tab_df:
    tab_df.write(data)

with tab_plot:
    tab_plot.line_chart(data)

with tab_decomp:
    tab_decomp.pyplot(decompose(data), use_container_width = True)

with tab_acf:
    tab_acf.pyplot(plot_acf(data), use_container_width = True)

with tab_pacf:
    tab_pacf.pyplot(plot_pacf(data), use_container_width = True)

with tab_adf:
    st.markdown('''
    Введем две гипотезы
    
    H0: Временной ряд является нестационарным. Другими словами, он имеет некоторую структуру, зависящую от времени, и не имеет постоянной дисперсии во времени.
    
    H1: Временной ряд является стационарным

    ''')
    pvalue = stattools.adfuller(data)[1]
    st.write(f'Значение p-value равно {pvalue:.3f}')
    st.write('Вывод: ', end = '')
    if pvalue < .01:
        st.write(f'Можно отвергуть нулевую гипотезу при уровне значимости 0.01, можно считать ряд стационарным')
    elif pvalue < .05:
        st.write(f'Можно отвергуть нулевую гипотезу при уровне значимости 0.05, можно считать ряд стационарным')
    else:
        st.write(f'Не удалось отвергнуть нулевую гипотезу, нельзя сказать о том, что ряд стационарный')
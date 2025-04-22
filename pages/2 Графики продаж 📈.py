import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import custom_module as gap
import os


@st.cache_data
def load_data():
    df = gap.load_df(file)
    return df


def get_hist_plot(data):
    fig, ax = plt.subplots()
    ax.hist(data)
    ax.legend(data.columns.tolist(), fontsize = 16)
    plt.title('Гистограмма количества продаж', fontsize = 16)
    return fig


st.set_page_config(page_title="История продаж",
                   layout = 'wide')

absolute_path = os.path.abspath(os.path.dirname('main_window.py'))
file = absolute_path + "/data/sales_plus.csv"

df = load_data()

articles = df.columns.tolist()
default_value = articles[0]
arts = st.multiselect('Список артикулов', articles, default_value)

tab_df, tab_plot, tab_hist = st.tabs(["🗃 Данные", "📈 История продаж", '📊 Гистограмма продаж'])
data = df[arts]

tab_df.write(data)

tab_plot.line_chart(data)


plt.style.use('seaborn-v0_8')

tab_hist.pyplot(get_hist_plot(data), use_container_width = True)

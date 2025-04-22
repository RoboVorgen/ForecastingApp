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


st.set_page_config(page_title="Таблица с данными",
                   layout = 'wide')

absolute_path = os.path.abspath(os.path.dirname('main_window.py'))
file = absolute_path + "/data/sales.csv"

df = load_data()

st.write(df)
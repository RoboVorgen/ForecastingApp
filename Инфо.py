import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def main():
  st.set_page_config(page_title="Главная страница",
                     layout = 'wide'
                    )
  st.set_option('client.toolbarMode', 'developer')
  st.set_option('client.showSidebarNavigation', True)
  st.write("## Данное web-приложение создано для работы с временными рядами")
  st.markdown("""
          ### На данный момент в приложении реализован следующий функционал:
      
          * Отображение графиков
          * Анализ временного ряда
          * Прогнозирование ряда:
              - наивными моделями
              - трендовыми моделями
              - моделями экспоненчиального сглаживания
              - моделями ARIMA
              - ~~моделями машинного обучения~~
              - ~~нейронными сетями~~
  """
  )

   

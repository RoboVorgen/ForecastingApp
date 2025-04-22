import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import (ExponentialSmoothing as ES,
                                         SimpleExpSmoothing as SES,
                                         Holt
)
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from pmdarima import AutoARIMA


import warnings
warnings.filterwarnings('ignore')


def linear_trend(timeseries):
    
    ts = timeseries.copy().reset_index()
    article = ts.columns.tolist()[1]
    ts['Дата'] = ts.index + 1
    train = ts[:-3]
    lr_model = LinearRegression(n_jobs=-1)
    lr_model.fit(train['Дата'].values.reshape(-1, 1), train[article].values.reshape(-1, 1))
    ts['тренд'] = lr_model.predict(ts['Дата'].values.reshape(-1,1))
    return ts['тренд'].values


def log_trend(timeseries):
    ts = timeseries.copy().reset_index()
    article = ts.columns.tolist()[1]
    ts['Дата'] = ts.index + 1
    ts['Дата'] = ts['Дата'].apply(lambda x: np.log10(x))
    train = ts[:-3]
    lr_model = LinearRegression(n_jobs=-1)
    lr_model.fit(train['Дата'].values.reshape(-1, 1),
                 train[article].values.reshape(-1, 1))
    ts['тренд'] = lr_model.predict(ts['Дата'].values.reshape(-1,1))
    return ts['тренд'].values


def hyperbolic_trend(timeseries):
    ts = timeseries.copy().reset_index()
    article = ts.columns.tolist()[1]
    ts['Дата'] = ts.index + 1
    ts['Дата'] = ts['Дата'].apply(lambda x: 1/(x+1))
    train = ts[:-3]
    lr_model = LinearRegression(n_jobs=-1)
    lr_model.fit(train['Дата'].values.reshape(-1, 1), train[article].values.reshape(-1, 1))
    ts['тренд'] = lr_model.predict(ts['Дата'].values.reshape(-1,1))
    return ts['тренд'].values


def additive_ts(timeseries):
    art = timeseries.columns[0]
    ts = timeseries[:-3].copy()
    ts['t'] = [i+1 for i in range(len(ts))]
    # ts = ts.set_index('Дата')
    ts['m'] = ts.index.month
    ts['y'] = ts.index.year
    ts['y'] -= ts['y'].min() - 1
    ts['roll_mean'] = ts[art].rolling(12).mean().shift(-6)
    ts['centred_roll_mean'] = ts['roll_mean'].rolling(2).mean()
    ts['Seasonal'] = ts[art] - ts['centred_roll_mean']
    seas = pd.DataFrame(data = ts['m'].sort_values().unique(), columns = ['m'])
    years = ts.loc[~ts['Seasonal'].isna(), 'y'].unique().tolist()
    for y in years:
        seas = seas.merge(ts.loc[ts['y'] == y, ['m','Seasonal']], left_on = 'm', right_on = 'm', how = 'left')
        seas = seas.rename(columns = {'Seasonal': y})
    seas['seas_mean'] = seas[years].mean(axis = 1)
    seas['S'] = seas['seas_mean'] - seas['seas_mean'].mean()
    
    final = timeseries.copy()
    final['t'] = [i+1 for i in range(len(final))]
    # final = final.set_index('Дата')
    final['m'] = final.index.month
    final = final.merge(seas[['m', 'S']], left_on = 'm', right_on = 'm', how = 'left')
    final['Y-S'] = final[art] - final['S']
    X_train = final[:-3]['t'].values.reshape(-1, 1)
    y_train = final[:-3]['Y-S'].values.reshape(-1, 1)
    lr_model = LinearRegression(n_jobs = -1)
    lr_model.fit(X_train, y_train)
    final['T'] = lr_model.predict(final['t'].values.reshape(-1, 1))
    final['Y'] = final['T'] + final['S']
    return final['Y'].values

def mean_prediction(timeseries):
    art = timeseries.columns[0]
    train = timeseries[:-3][art]
    ts = timeseries.copy()
    ts['Прогноз средним'] = train.mean()
    return ts['Прогноз средним'].values


def median_prediction(timeseries):
    art = timeseries.columns[0]
    train = timeseries[:-3][art]
    ts = timeseries.copy()
    ts['Прогноз медианой'] = train.median()
    return ts['Прогноз медианой'].values


def lag_prediction(timeseries):
    art = timeseries.columns[0]
    ts = timeseries.copy()
    ts['Прогноз предыдущим'] = ts[art].shift()
    return ts['Прогноз предыдущим'].values


def mean_seasonal_prediction(timeseries):
    art = timeseries.columns[0]
    ts = timeseries.copy()
    ts['Прогноз средним'] = ts[art].rolling(3).mean().shift(12-2)
    return ts['Прогноз средним'].values


def median_seasonal_prediction(timeseries):
    art = timeseries.columns[0]
    ts = timeseries.copy()
    ts['Прогноз медианой'] = ts[art].rolling(3).median().shift(12-2)
    return ts['Прогноз медианой'].values


def lag_seasonal_prediction(timeseries):
    art = timeseries.columns[0]
    ts = timeseries.copy()
    ts['Прогноз предыдущим'] = ts[art].shift(12)
    return ts['Прогноз предыдущим'].values


def auto_MA(timeseries):
    ts = timeseries.copy()
    art = ts.columns[0]
    train = ts[:-3]
    best_model = AutoARIMA(
        start_p = 0,
        max_p = 0,
        d = 0,
        max_d = 0,
        start_q = 0,
        max_q = 12,
        start_P = 0, 
        max_P = 0,
        D = 0,
        max_D = 0,
        start_Q = 0,
        max_Q = 0,
        max_order = None,
        maxiter=100,
        n_jobs = -1,
        n_fits = 100,
        random = False,
        information_criterion = 'hqic'
    )
    best_model.fit(train[art])
    pred = best_model.predict_in_sample().tolist()
    ts['Auto-MA'] = pred + best_model.predict(3).tolist()
    return ts['Auto-MA'].values


def auto_ARIMA(timeseries):
    ts = timeseries.copy()
    art = ts.columns[0]
    train = ts[:-3]
    best_model = AutoARIMA(
        start_p = 0,
        max_p = 12,
        d = 0,
        max_d = 4,
        start_q = 0,
        max_q = 12,
        start_P = 0, 
        max_P = 0,
        D = 0,
        max_D = 0,
        start_Q = 0,
        max_Q = 0,
        max_order = None,
        maxiter=1000,
        n_jobs = -1,
        n_fits = 100,
        random = False,
        information_criterion = 'aic'
    )
    best_model.fit(train[art])
    pred = best_model.predict_in_sample().tolist()
    ts['Auto-ARIMA'] = pred + best_model.predict(3).tolist()
    return ts['Auto-ARIMA'].values


def auto_SARIMAX(timeseries):
    ts = timeseries.copy()
    art = ts.columns[0]
    train = ts[:-3]
    best_model = AutoARIMA(
        start_p = 0,
        max_p = 12,
        d = 0,
        max_d = 4,
        start_q = 0,
        max_q = 12,
        start_P = 0, 
        max_P = 12,
        D = 0,
        max_D = 4,
        start_Q = 0,
        max_Q = 12,
        max_order = None,
        maxiter=1000,
        m = 12,
        n_jobs = -1,
        n_fits = 100,
        random = False,
        information_criterion = 'aic'
    )
    best_model.fit(train[art])
    pred = best_model.predict_in_sample().tolist()
    ts['Auto-SARIMAX'] = pred + best_model.predict(3).tolist()
    return ts['Auto-SARIMAX'].values


def SES_pred(timeseries):
    ts = timeseries.copy()
    art = ts.columns[0]
    train = ts[:-3]
    SES_model = SES(train[art]).fit()
    ts['SES'] = SES_model.predict(start = 0, end = len(timeseries) - 1)
    return ts['SES'].values


def ES_pred(timeseries):
    ts = timeseries.copy()
    art = ts.columns[0]
    train = ts[:-3]
    ES_model = ES(train[art], seasonal = 'add', seasonal_periods=12).fit()
    ts['ES'] = ES_model.predict(start = 0, end = len(timeseries) - 1)
    return ts['ES'].values


def Holt_pred(timeseries):
    ts = timeseries.copy()
    art = ts.columns[0]
    train = ts[:-3]
    Holt_model = Holt(train[art]).fit()
    ts['Holt'] = Holt_model.predict(start = 0, end = len(timeseries) - 1)
    return ts['Holt'].values


def ETS_pred(timeseries):
    ts = timeseries.copy()
    art = ts.columns[0]
    train = ts[:-3]
    ETS_model = ETSModel(train[art], trend = 'add', seasonal = 'add', seasonal_periods=12).fit(full_output = False, disp = False)
    ts['ETS'] = ETS_model.predict(start = 0, end = len(timeseries) - 1)
    return ts['ETS'].values
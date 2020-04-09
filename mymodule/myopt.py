import datetime
import pandas as pd
import talib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import pytz


def list_get(length: int, value=None) -> list:
    if length < 1:
        raise ValueError('The Length(%4d) should not lower than 1' % length)
    return [value for count in range(length)]


def list_fix_add(alist: list, x) -> list:
    if len(alist) <= 0:
        raise ValueError('The list is empty')
    alist.pop(0)
    alist.append(x)
    return alist


def get_odata(futudata: pd.DataFrame) -> pd.DataFrame:
    futudata.reset_index(drop=True, inplace=True)
    data = pd.DataFrame()
    data['time'] = futudata['time_key']
    data['open'] = futudata['open']
    data['close'] = futudata['close']
    data['high'] = futudata['high']
    data['low'] = futudata['low']
    data['turnover'] = futudata['turnover']
    return data


def get_linear_regression(close, predict=0):
    if close.ndim != 2:
        raise ValueError('input dimension ({}) != 2'.format(close.ndim))
    x = np.arange(0, len(close)).reshape(-1, 1)
    predict_x = np.arange(0, len(close) + predict).reshape(-1, 1)
    lr_reg = LinearRegression()
    lr_reg.fit(x, np.mean(close, axis=1))
    return lr_reg.predict(predict_x)


def get_polynomial_regression(close: np.ndarray, degree=2, predict=0, weight=None):
    if close.ndim != 2:
        raise ValueError('input dimension ({}) != 2'.format(close.ndim))
    poly_reg = PolynomialFeatures(degree=degree)
    x = np.arange(0, len(close)).reshape(-1, 1)
    predict_x = poly_reg.fit_transform(np.arange(0, len(close) + predict).reshape(-1, 1))
    poly_x = poly_reg.fit_transform(x)
    lr_reg = LinearRegression()
    lr_reg.fit(poly_x, np.mean(close, axis=1), sample_weight=weight)
    return lr_reg.predict(predict_x)


def fix_max_min(data, max_values: float, min_values: float):
    return (data - min_values) / (max_values - min_values)


def make_3min(o, s):
    t = np.zeros((int((s + 1) / 3) + (1 if (s + 1) % 3 != 0 else 0), 3))
    for i in range(2, s + 1, 3):
        close = o[i, 0]
        high = max(o[i - 3 + 1:i + 1, 1])
        low = min(o[i - 3 + 1:i + 1, 2])
        t[int(i / 3), 0:3] = np.array([close, high, low])
    if (s + 1) % 3 != 0:
        close = o[s, 0]
        high = max(o[s - s % 3:s + 1, 1])
        low = min(o[s - s % 3:s + 1, 2])
        t[int((s + 1) / 3), 0:3] = np.array([close, high, low])
    return t

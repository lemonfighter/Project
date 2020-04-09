import numpy as np
import pandas as pd
from talib import abstract
import datetime


class Regression:
    def __init__(self):
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import PolynomialFeatures
        self.Score = None
        self.Coef = None
        self.Model = LinearRegression
        self.Poly = PolynomialFeatures

    def predict(self, y, degree=1, predict=0, weight=None):
        if y.ndim != 2:
            raise ValueError('input dimension ({}) != 2'.format(y.ndim))

        poly_reg = self.Poly(degree)
        x = np.arange(0, len(y)).reshape(-1, 1)
        predict_x = poly_reg.fit_transform(np.arange(0, len(x) + predict).reshape(-1, 1))
        poly_i = poly_reg.fit_transform(x)
        lr_reg = self.Model()
        lr_reg.fit(poly_i, np.mean(y, axis=1), sample_weight=weight)
        self.Coef = np.array(np.concatenate(([lr_reg.intercept_], lr_reg.coef_[1:])))
        self.Score = lr_reg.score(poly_i, np.mean(y, axis=1))
        return lr_reg.predict(predict_x)


def build_ta(df: pd.DataFrame, f_p_n:list) -> pd.DataFrame:
    odata = df[['open', 'high', 'low', 'close', 'volume', 'time_key', 'change_rate', 'last_close']].copy()
    odata['volume'] = odata['volume'].astype('float64')
    odata = odata.reset_index(drop=True)
    if isinstance(f_p_n, list):
        for f, p, n in f_p_n:
            if isinstance(f, str):
                if f.upper() in abstract.__TA_FUNCTION_NAMES__:
                    if isinstance(p, tuple) or p is None:
                        try:
                            if p is None:
                                df = getattr(abstract, f.upper())(*(odata,))
                            else:
                                df = getattr(abstract, f.upper())(*((odata,) + p))
                        except Exception as e:
                            print(e)
                            msg = 'You can see the parameter input in http://mrjbq7.github.io/ta-lib/'
                            raise ValueError(msg)
                        if isinstance(df, pd.Series):
                            df = df.to_frame(n[0] if isinstance(n, list) else n)
                        else:
                            df.columns = list(n)
                        odata = pd.concat((odata, df), axis=1)
                    else:
                        msg = "Get type of param is {}, except 'tuple'".format(type(p))
                        raise ValueError(msg)
                else:
                    msg = "Do not have function {}, Ta function list :\n".format(f.upper())
                    for tan in abstract.__TA_FUNCTION_NAMES__:
                        msg += tan + '\n'
                    raise ValueError(msg)
            elif callable(f):
                if isinstance(p, tuple) or p is None:

                    try:
                        if p is None:
                            df = f(*(odata,))
                        else:
                            df = f(*((odata,) + p))
                    except Exception as e:
                        print(e)

                    if isinstance(df, pd.Series):
                        df = df.to_frame(n)
                    elif isinstance(df, pd.DataFrame):
                        df.columns = list(n)
                    else:
                        msg = "type of the output({}) is {}, expect pd.Serise or pd.DataFrame".format(f.__name__, type(df))
                        raise ValueError(msg)
                    for t in list(n):
                        if t in odata.columns:
                            odata = odata.drop(t, axis=1)
                    odata = pd.concat((odata, df), axis=1)
                else:
                    msg = "Get type of param is {}, except 'tuple'".format(type(p))
                    raise ValueError(msg)
            else:
                msg = "type of f is {}, except 'str'".format(type(f))
                raise ValueError(msg)

    elif f_p_n is not None:
        msg = "type of f_n_c is {}, expect 'list'\n".format(type(f_p_n))
        msg += "format : list(list(str(ta_function), tuple(param), list(str(column_name), ... )))\n"
        msg += "         [['sma', (14,), ('sma14')], ['stoch', None, ['K', 'D']]]"
        raise ValueError(msg)
    return odata


def time_cutter(df: pd.DataFrame, time_range=('00:00:00', '23:59:59')) -> pd.DataFrame:
    day = datetime.datetime.strptime(df.iloc[0]['time_key'], '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d ')
    t1 = day + time_range[0]
    t2 = day + time_range[1]
    df = df[df['time_key'] >= t1]
    df = df[df['time_key'] <= t2]
    return df


if __name__ == '__main__':
    x = 1

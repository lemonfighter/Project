import time
import pytz
import datetime


class FutuFunction:
    # Version 3.1
    def __init__(self, timeout_period=2):
        import futu
        self.Opt = futu
        print('==== FuTu Connected ====')
        time.sleep(0.01)
        self.connect = self.Opt.OpenQuoteContext(host='127.0.0.1', port=11111)
        self.TimeoutLoop = timeout_period

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _reconnect(self):
        self.connect.close()
        time.sleep(3)
        self.connect.start()

    def _action(self, action):
        count = 0
        while True:
            report = action
            count += 1
            if report[0] >= 0:
                break
            else:
                self._reconnect()
                if count >= self.TimeoutLoop != 0:
                    msg = 'Futu Connection Error\nError Report: {}'.format(report)
                    raise RuntimeError(msg)
        return report[1]

    def get_snapshot(self, stock_code: list):
        return self._action(self.connect.get_market_snapshot(stock_code))

    def get_marketstatus(self):
        return self._action(self.connect.get_global_state())

    def get_trade_calender(self, market, start_date="%Y-%m-%d", end_date="%Y-%m-%d") -> list:
        return self._action(self.connect.get_trading_days(self.Opt.Market.US if market is None else self.Opt.Market.HK, start_date, end_date))

    def get_trade_calender2(self, market, day_sample: int, sample_delay: int) -> list:
        calender = []
        if market is self.Opt.Market.HK or market is self.Opt.Market.HK_FUTURE:
            time_zone = pytz.timezone('Asia/Hong_Kong')
        elif market is self.Opt.Market.SH or market is self.Opt.Market.SZ:
            time_zone = pytz.timezone('Asia/Shanghai')
        elif market is self.Opt.Market.US:
            time_zone = pytz.timezone('America/New_York')
        else:
            msg = 'market number is ({}), use class.Opt.Market.market'.format(market)
            raise ValueError(msg)

        day_ratio = 1.5
        while len(calender) < sample_delay + day_sample:
            st = (datetime.datetime.now(time_zone) - datetime.timedelta(days=(day_sample + sample_delay) * day_ratio + 10)).strftime('%Y-%m-%d')
            et = (datetime.datetime.now(time_zone)).strftime('%Y-%m-%d')
            calender = self.get_trade_calender(market, st, et)
            day_ratio += 0.1

        return calender[-(day_sample + sample_delay):-sample_delay]

    def get_kline_data(self, stock: str, day_sample: int, sample_delay: int, kline_type, count=100000, to_list=False):
        day_ratio = 1.5
        calender = []

        # time zone
        if 'HK.' or 'HK_FUTURE.' in stock:
            time_zone = pytz.timezone('Asia/Hong_Kong')
            if 'HK_FUTURE.' in stock:
                market = self.Opt.Market.HK_FUTURE
            else:
                market = self.Opt.Market.HK
        elif 'SH.' in stock:
            time_zone = pytz.timezone('Asia/Shanghai')
            market = self.Opt.Market.SH
        elif 'SZ.' in stock:
            time_zone = pytz.timezone('Asia/Shanghai')
            market = self.Opt.Market.SZ
        elif 'US.' in stock:
            time_zone = pytz.timezone('America/New_York')
            market = self.Opt.Market.US
        else:
            msg = 'stock number is ({}), expect ({})'.format(stock, 'HK. ,HK_FUTURE., SH., SZ., US.')
            raise ValueError(msg)

        while len(calender) < sample_delay + day_sample:
            st = (datetime.datetime.now(time_zone) - datetime.timedelta(days=(day_sample + sample_delay) * day_ratio + 10)).strftime('%Y-%m-%d')
            et = (datetime.datetime.now(time_zone)).strftime('%Y-%m-%d')
            calender = self.get_trade_calender(market, st, et)
            day_ratio += 0.1

        st = calender[-(day_sample + sample_delay)]['time']
        et = calender[-sample_delay - 1]['time']

        df = self._action(self.connect.request_history_kline(stock, st, et, kline_type, max_count=count))

        if not to_list:
            return df

        alist = []
        for dicts in calender[-(day_sample + sample_delay):-sample_delay] if sample_delay > 0 else calender[-(day_sample + sample_delay):]:
            start_date = dicts['time'] + ' 00:00:00'
            end_date = dicts['time'] + ' 23:59:59'
            t = df[df['time_key'] >= start_date]
            alist.append(t[t['time_key'] <= end_date])
        return alist

    def us_date_2_hk(self, df, to_list=False):
        a = len(df)
        for i in range(len(df)):
            t = datetime.datetime.strptime(df.loc[i, 'time_key'], "%Y-%m-%d %H:%M:%S")
            df.loc[i, 'time_key'] = (t + datetime.timedelta(hours=12 if "09-30" >= t.strftime("%m-%d") >= "02-09" else 13)).strftime("%Y-%m-%d %H:%M:%S")
            print('\nTotal : {}, Now : {}'.format(a, i), end='')
        if not to_list:
            return df
        print('test')
        start = df.iloc[0]['time_key']
        end = df.iloc[-1]['time_key']
        t = []
        while start <= end:
            td = datetime.datetime.strptime(start, "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d")
            t2 = df[df['time_key'] >= td + ' 00:00:00']
            t3 = t2[t2['time_key'] <= td + ' 23:59:59']
            if len(t3) > 0:
                t.append(t3)
            start = (datetime.datetime.strptime(start, "%Y-%m-%d %H:%M:%S") + datetime.timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
        return t

    def close(self):
        self.connect.close()
        time.sleep(0.01)
        print('==== FuTu Disconnected ====')

    def restart(self):
        self.connect.start()
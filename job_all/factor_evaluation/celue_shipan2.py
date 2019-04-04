# coding=utf-8

import sys

from restapi.realdataapi import get_exsymbol_kline_multi
from utils.db_utils3 import upsert_strat_multi_symbol_signal

sys.path.append('..')
import pandas as pd
import numpy as np
import time
import talib as ta
import datetime
from utils.db_utils2 import upsert_longshort_signal_spot


def ts_sum(df, window=10):
    return df.rolling(window).sum()


def max_s(x, y):
    value_list = [a if a > b else b for a, b in zip(x, y)]
    return pd.Series(value_list, name="max")


def min_s(x, y):
    value_list = [a if a < b else b for a, b in zip(x, y)]
    return pd.Series(value_list, name="min")


def delay(df, period=1):
    return df.shift(period)


def sma(df, window=10):
    return df.rolling(window).mean()


class Alphas(object):
    def __init__(self, pn_data):
        """
        :传入参数 pn_data: pandas.Panel
        """
        # 获取历史数据
        self.open = pn_data['open']
        self.high = pn_data['high']
        self.low = pn_data['low']
        self.close = pn_data['close']
        self.volume = pn_data['volume']
        self.amount = pn_data['amount']
        self.returns = self.close-self.close.shift(1)

    def alpha018(self):
        return self.close/delay(self.close,5)

    def alpha050(self):
        data_mid1 = [0 if x >= y else z for x, y, z in zip((self.high + self.low), (delay(self.high) + delay(self.low)), (max_s((self.high - delay(self.high)).abs(), (self.low - delay(self.low)).abs())))]
        data_mid1 = pd.Series(data_mid1, name="values")
        data_mid1 = ts_sum(data_mid1, 12)

        data_mid2 = [0 if x <= y else z for x, y, z in zip((self.high + self.low), (delay(self.high) + delay(self.low)), (max_s((self.high - delay(self.high)).abs(), (self.low - delay(self.low)).abs())))]
        data_mid2 = pd.Series(data_mid2, name="values")
        data_mid2 = ts_sum(data_mid2, 12)

        data_mid3 = data_mid1/(data_mid1+data_mid2)

        data_mid4 = [0 if x <= y else z for x, y, z in zip((self.high + self.low), (delay(self.high) + delay(self.low)), (max_s((self.high - delay(self.high)).abs(), (self.low - delay(self.low)).abs())))]
        data_mid4 = pd.Series(data_mid4, name="values")
        data_mid4 = ts_sum(data_mid4, 12)

        data_mid5 = [0 if x >= y else z for x, y, z in zip((self.high + self.low), (delay(self.high) + delay(self.low)), (max_s((self.high - delay(self.high)).abs(), (self.low - delay(self.low)).abs())))]
        data_mid5 = pd.Series(data_mid5, name="values")
        data_mid5 = ts_sum(data_mid5, 12)

        data_mid6 = data_mid4/(data_mid4+data_mid5)

        return data_mid6-data_mid3

    def alpha052(self):
        data_mid1 = self.high-delay((self.high+self.low+self.close)/3)
        data_mid1[data_mid1 < 0] = 0
        data_mid2=delay((self.high+self.low+self.close)/3)-self.low
        data_mid2[data_mid2 < 0] = 0
        return ts_sum(data_mid1, 26)/ts_sum(data_mid2, 26)*100

    def alpha055(self):
        data_mid1 = (self.close-delay(self.close)+(self.close-self.open)/2+delay(self.close)-delay(self.open))*16

        data_mid_z = (self.high-delay(self.close)).abs()+(self.low-delay(self.close)).abs()/2+(delay(self.close)-delay(self.open)).abs()/4
        data_mid_vz = (self.low-delay(self.close)).abs()+(self.high-delay(self.close)).abs()/2+(delay(self.close)-delay(self.open)).abs()/4
        data_mid_vv = (self.high-delay(self.low)).abs()+(delay(self.close)-delay(self.open))/4

        data_mid_v = [vz if x1>y1 and x2>y2 else vv for x1,y1,x2,y2,vz,vv in zip((self.low-delay(self.close)).abs(),(self.high-delay(self.low)).abs(),(self.low-delay(self.close)).abs(),(self.high-delay(self.close)).abs(),data_mid_vz,data_mid_vv)]
        data_mid2 = [z if x1>y1 and x2>y2 else v for x1,y1,x2,y2,z,v in zip((self.high-delay(self.close)).abs(),(self.low-delay(self.close)).abs(),(self.high-delay(self.close)).abs(),(self.high-delay(self.low)).abs(),data_mid_z,data_mid_v)]

        data_mid3 = max_s((self.high-delay(self.close)).abs(),(self.low-delay(self.close)).abs())

        data_all = data_mid1/data_mid2*data_mid3

        return ts_sum(data_all, 20)

    def alpha060(self):
        data_mid1=((self.close-self.low)-(self.high-self.close))/(self.high-self.low)
        return ts_sum(data_mid1,20)

    def alpha069(self):
        dtm = [0 if x <= y else z for x, y, z in zip(self.open, delay(self.open), max_s((self.high-self.open), (self.open-delay(self.open))))]
        dbm = [0 if x >= y else z for x, y, z in zip(self.open, delay(self.open), max_s((self.open-self.low), (self.open - delay(self.open))))]
        dtm = pd.Series(dtm, name="dtm")
        dbm = pd.Series(dbm, name="dbm")
        data_mid_z = (ts_sum(dtm, 20)-ts_sum(dbm, 20))/ts_sum(dtm, 20)
        data_mid_vz = (ts_sum(dtm, 20)-ts_sum(dbm, 20))/ts_sum(dbm, 20)

        data_mid_v = [0 if x == y else z for x, y, z in zip(ts_sum(dtm, 20), ts_sum(dbm, 20), data_mid_vz)]
        data_mid = [z if x > y else v for x, y, z, v in zip(ts_sum(dtm, 20), ts_sum(dbm, 20), data_mid_z, data_mid_v)]

        return pd.Series(data_mid, name="values")

    def alpha071(self):
        return (self.close-sma(self.close, 24))/sma(self.close, 24)*100


symbols = ["ethbtc", "xrpbtc", "mdabtc", "eosbtc", "xlmbtc", "tusdbtc", "ltcbtc",
           "stratbtc", "trxbtc", "adabtc", "iotabtc", "xmrbtc", "bnbbtc", "dashbtc",
           "xembtc", "etcbtc", "neobtc", "ontbtc", "zecbtc", "wavesbtc", "btgbtc",
           "vetbtc", "qtumbtc", "omgbtc", "zrxbtc", "gvtbtc", "bchabcbtc", "bchsvbtc"]

cols = ["period", "symbol", "tickid", "open", "high", "low", "close", "volume", "amount"]

slippage = 0.002

alpha_test = ["Alpha.alpha018", "Alpha.alpha050", "Alpha.alpha052", "Alpha.alpha052",
              "Alpha.alpha055", "Alpha.alpha060", "Alpha.alpha069", "Alpha.alpha071"]


def get_max_coin(now_time, df_all, alpha=alpha_test):
    end_time = now_time - now_time % 14400
    df_all = df_all[df_all["tickid"] < end_time]
    results = []
    for i in range(len(alpha)):
        result_dict_temp = {}
        for symbol in symbols:
            df_temp = df_all[df_all["symbol"] == symbol].copy()
            if len(df_temp) == 0:
                pass
            else:
                df_temp.index = range(len(df_temp))
                Alpha = Alphas(df_temp)
                df_temp[alpha[i]] = eval(alpha[i])()
                result_dict_temp[symbol] = df_temp[alpha[i]].dropna().values[-1]
        results.append(result_dict_temp)
    result_dict_last = {}
    for _ in results:
        for k, v in _.items():
            result_dict_last.setdefault(k, []).append(v)
    df = pd.DataFrame.from_dict(result_dict_last)
    print(df)
    df.to_csv('factors.csv')
    df = df.rank(axis=1, numeric_only=True, na_option="keep")
    series = df.sum()
    max_symble = series.idxmax()
    return max_symble


# 获取币对当前的价格
def get_now_price(now_time, coin, df_all):
    last_4h_time = now_time - now_time % 14400 - 14400
    df_all = df_all[(df_all["tickid"] == last_4h_time) & (df_all["symbol"] == coin)]
    now_price = df_all["close"].values[0]
    return now_price


# 获取币对当前25日均价
def get_now_ma25(now_time, df_all, coin, ma=25):
    end_time = now_time-now_time % 14400
    df_all = df_all[(df_all["tickid"] < end_time) & (df_all["symbol"] == coin)]
    ma = ta.MA(df_all["close"].values, timeperiod=ma)[-1]
    print(ma)
    return ma


# 获取上一次策略运行之后的结果
def get_last_result():
    """
    这个函数的目的是得到策略上一次运行的结果，在写策略提醒的时候，我通过更新csv文件的方式保留每一次策略运行后的结果，
    在写实盘的时候也可以采用其他方式。
    :return: 策略的现金数（btc数目）、持仓币对、持仓币对数目、策略净值
    """
    data_result = pd.read_csv("/Users/wuyong/alldata/original_data/last_result.csv", index_col=0)
    cash = data_result.tail(1)["cash"].values[0]
    coin = data_result.tail(1)["coin"].values[0]
    coin_num = data_result.tail(1)["coin_num"].values[0]
    asset = data_result.tail(1)["asset"].values[0]
    return cash, coin, coin_num, asset


def multi_factor_new(now_time, df_all):
    max_coin = get_max_coin(now_time, df_all)  # 得到当前因子值最大的币对
    print("此时因子值最大的币对为：%s" % max_coin)
    starttime = now_time - now_time % 14400
    endtime = starttime + 14400
    stratid = 402
    exchange = "BIAN"
    period = "4hour"
    res_ls = []

    now_price = get_now_price(now_time, max_coin, df_all)

    # 当最大因子为目标币对并且实时价格大于25日均价时，valid会等于1
    if max_coin in ["ethbtc", "eosbtc", "xrpbtc", "trxbtc", "tusdbtc", "bchabcbtc", "bchsvbtc", "ontbtc", "ltcbtc",  "adabtc", "bnbbtc"]:  # 如果现在因子值最大的币对在目标币对当中
        coin_ma = get_now_ma25(now_time, df_all, max_coin)  # 得到该因子值最大币对最近的25日均价

        valid = 1 if now_price > coin_ma else 0  # 是否因子值最大的币对的实时价格大于它最近的25日均价
        upsert_strat_multi_symbol_signal(stratid, exchange, period, starttime, endtime, max_coin, now_price, valid)
        res_ls.append([stratid, exchange, period, starttime, endtime, max_coin, now_price, valid])

    else:
        # 如果这个因子值最大的币对不在在目标币对当中
        upsert_strat_multi_symbol_signal(stratid, exchange, period, starttime, endtime, max_coin, now_price, 0)
        res_ls.append([stratid, exchange, period, starttime, endtime, max_coin, now_price, 0])

    # 后端处理逻辑：
    # 如果空仓，且valid==1 则买入；
    # 如果持仓，判断最大因子的币对跟持仓币对是否一样，若一样，则保持；若不一样，则卖出，若不一样且valid==1，则买入新的最大因子的币对。

    print("=========================")
    columns = ['stratid', 'exchange', 'period', 'starttime', 'endtime', 'symbol', 'price', 'valid']
    df = pd.DataFrame(res_ls, columns=columns)
    print(df)


def run_multi_factor_402():
    # 获取当前时间，这里的时间戳做了处理，去掉了秒数转换成整分
    now_time = int(time.time())
    time_str_tmp = time.strftime('%Y-%m-%d %H:%M', time.localtime(now_time))
    now_time = int(time.mktime(time.strptime(time_str_tmp, '%Y-%m-%d %H:%M')))

    # 一次性获取所有币对的所需数据
    start_time = (now_time - now_time % 14400) - 35 * 14400
    start_time_str = time.strftime('%Y-%m-%d %H:%M', time.localtime(start_time))

    symbol_lst = ','.join(symbols)
    errcode, errmsg, df_all = get_exsymbol_kline_multi('BIAN', symbol_lst, "4hour", start_time_str, time_str_tmp)
    if errcode != 0:
        print("查询K线返回失败,errmsg="+errmsg)
        return

    df_all[["open", "close", "high", "low", "volume", "amount"]] = df_all[["open", "close", "high", "low", "volume", "amount"]].astype("float")
    multi_factor_new(now_time, df_all)


if __name__ == '__main__':
    run_multi_factor_402()








































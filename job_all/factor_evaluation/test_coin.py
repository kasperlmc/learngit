# coding=utf-8

import sys
sys.path.append('..')
import pandas as pd
import numpy as np
from lib.realtime_kline_api_all import get_realtime_data
import time
import talib as ta
import datetime
# from utils.db_utils2 import upsert_longshort_signal_spot


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

    def alpha003(self):
        data_mid1 = min_s(self.low, delay(self.close, 1))
        data_mid2 = max_s(self.high, delay(self.close, 1))
        data_mid3 = [z if x > y else v for x, y, z, v in zip(self.close, delay(self.close, 1), data_mid1, data_mid2)]
        data_mid3 = np.array(data_mid3)
        data_mid4 = self.close-data_mid3
        data_mid5 = [0 if x == y else z for x, y, z in zip(self.close, delay(self.close, 1), data_mid4)]
        data_mid5 = np.array(data_mid5)
        df = pd.Series(data_mid5, name="value")
        return ts_sum(df, 6)

    def alpha014(self):
        return self.close-delay(self.close, 5)

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

    def alpha051(self):
        data_mid4 = [0 if x <= y else z for x, y, z in zip((self.high + self.low), (delay(self.high) + delay(self.low)), (max_s((self.high - delay(self.high)).abs(), (self.low - delay(self.low)).abs())))]
        data_mid4 = pd.Series(data_mid4, name="values")
        data_mid4 = ts_sum(data_mid4, 12)

        data_mid5 = [0 if x >= y else z for x, y, z in zip((self.high + self.low), (delay(self.high) + delay(self.low)), (max_s((self.high - delay(self.high)).abs(), (self.low - delay(self.low)).abs())))]
        data_mid5 = pd.Series(data_mid5, name="values")
        data_mid5 = ts_sum(data_mid5, 12)

        data_mid6 = data_mid4 / (data_mid4 + data_mid5)

        return data_mid6

    def alpha069(self):
        dtm = [0 if x <= y else z for x, y, z in zip(self.open, delay(self.open), max_s((self.high-self.open), (self.open-delay(self.open))))]
        dbm = [0 if x >= y else z for x, y, z in zip(self.open, delay(self.open), max_s((self.open-self.low), (self.open-delay(self.open))))]
        dtm = pd.Series(dtm, name="dtm")
        dbm = pd.Series(dbm, name="dbm")
        data_mid_z = (ts_sum(dtm, 20)-ts_sum(dbm, 20))/ts_sum(dtm, 20)
        data_mid_vz = (ts_sum(dtm, 20)-ts_sum(dbm, 20))/ts_sum(dbm, 20)

        data_mid_v = [0 if x == y else z for x, y, z in zip(ts_sum(dtm, 20), ts_sum(dbm, 20), data_mid_vz)]
        data_mid = [z if x > y else v for x, y, z, v in zip(ts_sum(dtm, 20), ts_sum(dbm, 20), data_mid_z, data_mid_v)]

        return pd.Series(data_mid, name="values")

    def alpha128(self):
        data_mid1 = (self.high+self.low+self.close)/3*self.volume
        data_mid1[(self.high+self.low+self.close)/3 <= delay((self.high+self.low+self.close)/3)] = 0
        data_mid2 = (self.high+self.low+self.close)/3*self.volume
        data_mid2[(self.high+self.low+self.close)/3 >= delay((self.high+self.low+self.close)/3)] = 0
        return 100-(100/(1+ts_sum(data_mid1, 14)/ts_sum(data_mid2, 14)))

    def alpha167(self):
        data_mid = self.close-delay(self.close)
        data_mid[self.close <= delay(self.close)] = 0
        return ts_sum(data_mid, 12)/self.close

    def alpha175(self):
        return sma(max_s(max_s((self.high-self.low), (delay(self.close)-self.high).abs()), (delay(self.close)-self.low).abs()), 6)/self.close


symbols = ["ethbtc", "xrpbtc", "mdabtc", "eosbtc", "xlmbtc", "tusdbtc", "ltcbtc",
           "stratbtc", "trxbtc", "adabtc", "iotabtc", "xmrbtc", "bnbbtc", "dashbtc",
           "xembtc", "etcbtc", "neobtc", "ontbtc", "zecbtc", "wavesbtc", "btgbtc",
           "vetbtc", "qtumbtc", "omgbtc", "zrxbtc", "gvtbtc", "bchabcbtc", "bchsvbtc"]

cols = ["exchange", "period", "symbol", "tickid", "open", "high", "low", "close", "volume", "amount"]

slippage = 0.002

alpha_test = ["Alpha.alpha003", "Alpha.alpha014", "Alpha.alpha050", "Alpha.alpha051",
              "Alpha.alpha069", "Alpha.alpha128", "Alpha.alpha167", "Alpha.alpha175"]


def get_max_coin(now_time, df_all, alpha=alpha_test):
    end_time = now_time - now_time % 14400
    df_all = df_all[df_all["tickid"] < end_time]
    results = []
    for i in range(len(alpha)):
        result_dict_temp = {}
        for symbol in symbols:
            df_temp = df_all[df_all["symbol"] == symbol]
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
    df = df.rank(axis=1, numeric_only=True, na_option="keep")
    series = df.sum()
    max_symble = series.idxmax()
    return max_symble


# 获取币对当前的价格
def get_now_price(now_time, coin, df_all):
    last_4h_time = now_time - now_time % 14400
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


# 依据各个币对的因子排名结果进行交易，以下为策略进行单次交易的交易函数
def multi_factor(now_time):
    max_coin = get_max_coin(now_time, df_all)  # 得到当前因子值最大的币对
    print("此时因子值最大的币对为：%s" % max_coin)
    last_4h = now_time - 14400
    starttime = now_time - now_time % 14400
    endtime = starttime + 14400
    coin = get_max_coin(last_4h, df_all)
    stratid = 13
    exchange = "BIAN"
    period = "4hour"
    optype = 2

    if max_coin in ["ethbtc", "eosbtc", "xrpbtc", "trxbtc", "tusdbtc", "bchabcbtc", "bchsvbtc", "ontbtc", "ltcbtc", "adabtc", "bnbbtc"]:  # 如果现在因子值最大的币对在目标币对当中

        coin_ma = get_now_ma25(now_time, df_all, max_coin)  # 得到该因子值最大币对最近的25日均价

        if coin == max_coin:  # 如果之前因子值最大的币对和现在因子值最大的币对一样，则策略不动
            print("策略持仓不动")

        else:  # 如果之前因子值最大的币对不同于现在因子值最大的币对

            if coin in ["ethbtc", "eosbtc", "xrpbtc", "trxbtc","tusdbtc", "bchabcbtc", "bchsvbtc", "ontbtc", "ltcbtc", "adabtc", "bnbbtc"]:  # 如果之前因子值最大的币对属于目标币对

                name = coin
                dir = 1
                now_price = get_now_price(now_time, coin, df_all)
                # upsert_longshort_signal_spot(stratid, exchange, name, period, optype, starttime, endtime, dir, now_price)  # 往库中写入要卖出的币对

                if get_now_price(now_time, max_coin, df_all) > coin_ma:  # 如果现在因子值最大的币对的实时价格大于它最近的25日均价

                    print("策略转仓")
                    print("之前持有币对为：%s，现在持仓币对为：%s" % (coin, max_coin))
                    name = max_coin
                    dir = 0
                    now_price = get_now_price(now_time, max_coin, df_all)
                    # upsert_longshort_signal_spot(stratid, exchange, name, period, optype, starttime, endtime, dir, now_price)  # 往库中写入要买入的币对

                else:  # 如果如果因子值最大的币对的实时价格不大于它最近的25日均价，仅卖出此前策略持仓，而不买入因子值最大的币对

                    print("该币对当前不宜开仓，策略平仓，此前持仓币对为：%s" % coin)

            else:  # 如果之前因子值最大的币对不属于目标币对

                if get_now_price(now_time, max_coin, df_all) > coin_ma:  # 如果因子值最大的币对的实时价格大于它最近的25日均价

                    print("策略开仓")
                    print("本次开仓买入币对为：%s" % max_coin)
                    name = max_coin
                    dir = 0
                    now_price = get_now_price(now_time, max_coin, df_all)
                    # upsert_longshort_signal_spot(stratid, exchange, name, period, optype, starttime, endtime, dir, now_price)  # 往库中写入要买入的币对

                else:  # 如果因子值最大的币对的实时价格不大于它最近的25日均价，则策略不进行任何操作
                    print("该目标币对当前不宜开仓")

    else:  # 如果这个因子值最大的币对不在在目标币对当中
        if coin in ["ethbtc", "eosbtc", "xrpbtc", "trxbtc","tusdbtc", "bchabcbtc", "bchsvbtc", "ontbtc", "ltcbtc", "adabtc", "bnbbtc"]:  # 如果之前因子值最大的币对属于目标币对
            print("策略平仓")
            print("本次平仓卖出币对为：%s" % coin)
            name = coin
            dir = 1
            now_price = get_now_price(now_time, coin, df_all)
            # upsert_longshort_signal_spot(stratid, exchange, name, period, optype, starttime, endtime, dir, now_price)  # 往库中写入要卖出的币对

        else:  # 如果之前因子值最大的币对不属于目标币对，则策略不动
            print("策略不进行操作")


# 获取当前时间，这里的时间戳做了处理，去掉了秒数转换成整分
now_time = int(time.time())
time_str_tmp = time.strftime('%Y-%m-%d %H:%M', time.localtime(now_time))
now_time = int(time.mktime(time.strptime(time_str_tmp, '%Y-%m-%d %H:%M')))

# 一次性获取所有币对的所需数据
start_time = (now_time - now_time % 14400) - 100 * 14400
time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))
symbol_upper = [symbol.upper() for symbol in symbols]
df_all = get_realtime_data('BIAN', '4h', symbol_upper, start_time=time_str, end_time=None)
df_all = pd.DataFrame(df_all, columns=cols)
df_all[["open", "close", "high", "low", "volume", "amount"]] = df_all[["open", "close", "high", "low", "volume", "amount"]].astype("float")
print("get all data........")

for i in range(50):
    print(i)
    end_time = now_time - i*14400
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time-end_time % 14400)))
    print(get_max_coin(end_time, df_all))





































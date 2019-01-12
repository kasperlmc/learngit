# coding=utf-8

import sys
sys.path.append('..')
import numpy as np
import pandas as pd
from numpy import abs
from numpy import log
from numpy import sign
from scipy.stats import rankdata
from lib.myfun import *
import warnings
import copy
import statsmodels.api as sm
warnings.filterwarnings("ignore")

#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)

exchange = 'BITFINEX'
symbols = ["btcusdt", "ethusdt", "xrpusdt", "trxusdt", "eosusdt", "zecusdt", "ltcusdt",
           "etcusdt", "etpusdt", "iotusdt", "rrtusdt", "xmrusdt", "dshusdt", "avtusdt",
           "omgusdt", "sanusdt", "qtmusdt", "edousdt", "btgusdt", "neousdt", "zrxusdt",
           "tnbusdt", "funusdt", "mnausdt", "sntusdt", "gntusdt"]


for i in range(len(symbols)):
    dataf = read_data(exchange, symbols[i], '4h', "2017-01-01", "2018-12-21")
    # print(symbols[i])
    dataf["open_" + symbols[i]] = dataf["open"]
    dataf["close_" + symbols[i]] = dataf["close"]
    dataf["high_" + symbols[i]] = dataf["high"]
    dataf["low_" + symbols[i]] = dataf["low"]
    dataf["volume_" + symbols[i]] = dataf["volume"]

    if i == 0:
        data_open = dataf[["open_"+symbols[i],"date"]]
        data_close = dataf[["close_" + symbols[i], "date"]]
        data_high = dataf[["high_" + symbols[i], "date"]]
        data_low = dataf[["low_" + symbols[i], "date"]]
        data_volume = dataf[["volume_" + symbols[i], "date"]]

    else:
        data_open = data_open.merge(dataf[["open_"+symbols[i], "date"]], how="left",on="date")
        data_close = data_close.merge(dataf[["close_"+symbols[i], "date"]], how="left",on="date")
        data_high = data_high.merge(dataf[["high_" + symbols[i], "date"]], how="left", on="date")
        data_low = data_low.merge(dataf[["low_" + symbols[i], "date"]], how="left", on="date")
        data_volume = data_volume.merge(dataf[["volume_" + symbols[i], "date"]], how="left", on="date")


def rolling_rank(na):
    return rankdata(na)[-1]


def ts_rank(df,window=10):
    return window+1-df.rolling(window).apply(rolling_rank)


data_list = [data_open, data_high, data_low, data_close, data_volume]


class Alphas(object):
    def __init__(self, pn_data):
        """
        :传入参数 pn_data: pandas.Panel
        """
        # 获取历史数据
        self.date = pn_data[0]["date"].values
        self.open = pn_data[0]
        del self.open["date"]
        self.high = pn_data[1]
        del self.high["date"]
        self.low = pn_data[2]
        del self.low["date"]
        self.close = pn_data[3]
        del self.close["date"]
        self.volume = pn_data[4]
        del self.volume["date"]
        self.returns = pd.DataFrame((self.close.values/self.close.shift(1).values)-1, columns=self.close.columns)

    def alpha001(self):
        data_mid1 = self.volume.apply(lambda x: np.log(x))
        data_mid1 = data_mid1.diff()
        data_mid1 = data_mid1.rank(axis=1,numeric_only=True,na_option="keep")
        data_mid2 = pd.DataFrame((self.close.values-self.open.values)/self.open.values, columns=self.volume.columns)
        data_mid2 = data_mid2.rank(axis=1,numeric_only=True,na_option="keep")
        result = data_mid2.rolling(window=6).corr(data_mid1)*(-1)
        result["date"] = self.date
        result.columns = symbols+["date"]
        return result

    def alpha002(self):
        data_mid1 = pd.DataFrame((self.open.values*0.85+self.high.values*0.15), columns=self.open.columns)
        data_mid1 = data_mid1.diff(4)
        data_mid1 = data_mid1.apply(lambda x: np.sign(x))
        data_mid1 = data_mid1.rank(axis=1,numeric_only=True,na_option="keep")*(-1)
        data_mid1["date"] = self.date
        data_mid1.columns = symbols+["date"]
        return data_mid1

    def alpha003(self):
        data_mid1 = self.high.rank(axis=1,numeric_only=True,na_option="keep")
        data_mid2 = pd.DataFrame(self.volume.values, columns=self.high.columns).rank(axis=1,numeric_only=True,na_option="keep")
        result = data_mid2.rolling(window=10).corr(data_mid1)
        result = result.rank(axis=1,numeric_only=True,na_option="keep").rolling(window=10).sum()*-1
        result["date"] = self.date
        result.columns = symbols+["date"]
        return result

    def alpha004(self):
        data_mid1 = ts_rank(self.volume,window=5)
        data_mid2 = -1*(self.low.rolling(window=5).min())+(self.low.rolling(window=5).min()).shift(5)
        data_mid3 = ((self.returns.rolling(window=240).sum()-self.returns.rolling(window=20).sum())/220).rank(axis=1,numeric_only=True,na_option="keep")
        result = pd.DataFrame(data_mid1.values*data_mid2.values*data_mid3.values, columns=self.close.columns)
        result["date"] = self.date
        result.columns = symbols+["date"]
        return result

    def alpha005(self):
        data_mid1 = self.open.rolling(window=5).sum()
        data_mid2 = self.returns.rolling(window=5).sum()
        data_mid3 = pd.DataFrame(data_mid1.values*data_mid2.values, columns=self.close.columns)
        data_mid3 = (data_mid3-data_mid3.shift(10)).rank(axis=1,numeric_only=True,na_option="keep")*-1
        data_mid3["date"] = self.date
        data_mid3.columns = symbols+["date"]
        return data_mid3

    def alpha006(self):
        data_mid1 = self.high.rolling(window=10).std().rank(axis=1,numeric_only=True,na_option="keep")*-1
        data_mid2 = pd.DataFrame(self.volume.values, columns=self.high.columns)
        data_mid2 = self.high.rolling(window=10).corr(data_mid2)
        result = data_mid1*data_mid2
        result["date"] = self.date
        result.columns = symbols+["date"]
        return result

    def alpha007(self):
        data_mid1 = (self.close-self.close.shift(1)).apply(lambda x: np.sign(x))+\
                    (self.close.shift(1)-self.close.shift(2)).apply(lambda x: np.sign(x))+\
                    (self.close.shift(2)-self.close.shift(3)).apply(lambda x: np.sign(x))
        data_mid1 = data_mid1.rank(axis=1,numeric_only=True,na_option="keep")
        data_mid2 = pd.DataFrame(self.volume.values, columns=self.close.columns)
        data_mid2 = data_mid2.rolling(window=5).sum()*data_mid1
        data_mid3 = pd.DataFrame(self.volume.values, columns=self.close.columns).rolling(window=20).sum()
        result = -1*data_mid2/data_mid3
        result["date"] = self.date
        result.columns = symbols+["date"]
        return result

    def alpha008(self):
        data_mid1 = pd.DataFrame(self.open.values, columns=self.close.columns)
        data_mid2 = ((self.close-data_mid1).apply(lambda x: np.abs(x))).rolling(window=10).std()
        data_mid3 = self.close-data_mid1
        data_mid4 = self.close.rolling(window=10).corr(data_mid1)
        result = -1*(data_mid2+data_mid3+data_mid4).rank(axis=1,numeric_only=True,na_option="keep")
        result["date"] = self.date
        result.columns = symbols+["date"]
        return result

    def alpha009(self):
        data_mid1 = self.volume.rank(axis=1,numeric_only=True,na_option="keep")
        data_mid1 = pd.DataFrame(data_mid1.values, columns=self.high.columns)
        result = -1*(data_mid1.rolling(window=10).corr(self.high))
        result["date"] = self.date
        result.columns = symbols+["date"]
        return result

    def alpha010(self):
        data_mid1 = pd.DataFrame(self.volume.rank(axis=1,numeric_only=True,na_option="keep").values, columns=self.high.columns)
        data_mid2 = self.high.rank(axis=1,numeric_only=True,na_option="keep")
        result = data_mid1.rolling(window=5).cov(data_mid2).rank(axis=1,numeric_only=True,na_option="keep")*-1
        result["date"] = self.date
        result.columns = symbols+["date"]
        return result

    def alpha011(self):
        data_mid1 = (self.close-self.close.rolling(window=5).max()).rank(axis=1,numeric_only=True,na_option="keep")
        data_mid2 = pd.DataFrame(self.volume.rolling(window=40).mean().values, columns=self.low.columns)
        data_mid2 = data_mid2.rolling(window=5).corr(self.low).rank(axis=1,numeric_only=True,na_option="keep")
        result = data_mid1*pd.DataFrame(data_mid2.values, columns=self.close.columns)*-1
        result["date"] = self.date
        result.columns = symbols+["date"]
        return result

    def alpha012(self):
        data_mid1 = pd.DataFrame(self.volume.values, columns=self.close.columns).rank(axis=1,numeric_only=True,na_option="keep")
        data_mid2 = ((self.close.rank(axis=1,numeric_only=True,na_option="keep")).rolling(window=5).cov(data_mid1)).rank(axis=1,numeric_only=True,na_option="keep")*-1
        data_mid2["date"] = self.date
        data_mid2.columns = symbols+["date"]
        return data_mid2

    def alpha013(self):
        data_mid1 = self.high.rolling(window=5).corr(pd.DataFrame(self.volume.values, columns=self.high.columns))
        data_mid1 = data_mid1-data_mid1.shift(5)
        data_mid2 = (self.close.rolling(window=20).std()).rank(axis=1,numeric_only=True,na_option="keep")
        result = -1*(pd.DataFrame(data_mid1.values*data_mid2.values, columns=self.close.columns))
        result["date"] = self.date
        result.columns = symbols+["date"]
        return result

    def alpha014(self):
        data_mid1 = self.open.rank(axis=1,numeric_only=True,na_option="keep")
        data_mid2 = (pd.DataFrame(self.volume.values, columns=self.open.columns)).rank(axis=1,numeric_only=True,na_option="keep")
        result = data_mid1.rolling(window=10).corr(data_mid2)*-1
        result["date"] = self.date
        result.columns = symbols+["date"]
        return result

    def alpha015(self):
        data_mid1 = (self.open-pd.DataFrame(self.high.values, columns=self.open.columns).shift(1)).rank(axis=1,numeric_only=True,na_option="keep")
        data_mid2 = (self.open-pd.DataFrame(self.close.values, columns=self.open.columns).shift(1)).rank(axis=1,numeric_only=True,na_option="keep")
        data_mid3 = (self.open-pd.DataFrame(self.low.values, columns=self.open.columns).shift(1)).rank(axis=1,numeric_only=True,na_option="keep")
        result = -1*data_mid1*data_mid2*data_mid3
        result["date"] = self.date
        result.columns = symbols+["date"]
        return result

    def alpha016(self):
        data_mid1 = self.open.rolling(window=10).corr(pd.DataFrame(self.volume.values, columns=self.open.columns))
        data_mid2 = -1*(self.returns - self.returns.shift(3)).rank(axis=1,numeric_only=True,na_option="keep")
        result = pd.DataFrame(data_mid1.values*data_mid2.values, columns=self.close.columns)
        result["date"] = self.date
        result.columns = symbols+["date"]
        return result

    def alpha017(self):
        data_mid1 = (self.volume.rolling(window=15).mean()).rank(axis=1,numeric_only=True,na_option="keep")
        data_mid2 = self.high.rank(axis=1,numeric_only=True,na_option="keep")
        data_mid3 = data_mid2.rolling(window=9).corr(pd.DataFrame(data_mid1.values, columns=self.high.columns))
        result = data_mid3.rank(axis=1,numeric_only=True,na_option="keep")*-1
        result["date"] = self.date
        result.columns = symbols+["date"]
        return result

    def alpha018(self):
        data_mid1 = (ts_rank(self.close, window=10)).rank(axis=1,numeric_only=True,na_option="keep")*-1
        data_mid2 = ((self.close.diff()).diff()).rank(axis=1,numeric_only=True,na_option="keep")
        data_mid3 = (ts_rank(self.volume/(self.volume.rolling(window=20).mean()), window=5)).rank(axis=1,numeric_only=True,na_option="keep")
        result = pd.DataFrame(data_mid1.values*data_mid2.values*data_mid3.values, columns=self.close.columns)
        result["date"] = self.date
        result.columns = symbols+["date"]
        return result

    def alpha019(self):
        data_mid1 = self.close-pd.DataFrame(self.low.values, columns=self.close.columns).rolling(window=12).min()
        data_mid2 = (pd.DataFrame(self.high.values, columns=self.close.columns)).rolling(window=12).max()-(pd.DataFrame(self.low.values, columns=self.close.columns)).rolling(window=12).min()
        data_mid3 = (data_mid1/data_mid2).rank(axis=1,numeric_only=True,na_option="keep")
        data_mid4 = (pd.DataFrame(self.volume.values, columns=self.close.columns)).rank(axis=1,numeric_only=True,na_option="keep")
        result = data_mid3.rolling(window=6).corr(data_mid4)
        result["date"] = self.date
        result.columns = symbols+["date"]
        return result

    def alpha020(self):
        data_mid1 = (pd.DataFrame(self.open.values, columns=self.close.columns)-self.close).shift(1)
        data_mid1 = (data_mid1.rolling(window=200).corr(self.close)).rank(axis=1,numeric_only=True,na_option="keep")
        data_mid2 = (pd.DataFrame(self.open.values, columns=self.close.columns)-self.close).rank(axis=1,numeric_only=True,na_option="keep")
        result = data_mid1+data_mid2
        result["date"] = self.date
        result.columns = symbols+["date"]
        return result

    def alpha021(self):
        data_mid1 = 1-(pd.DataFrame(self.open.values, columns=self.close.columns)/self.close)
        data_mid1 = (data_mid1**2)*-1
        result = data_mid1.rank(axis=1, numeric_only=True, na_option="keep")
        result["date"] = self.date
        result.columns = symbols+["date"]
        return result


Alpha = Alphas(data_list)
# print(Alpha.alpha021().head(30))
data_result = Alpha.alpha021()
# print(data_result.head(10))
# print(type(data_result["date"].values[0]))

factor_name = "Alpha.alpha222"
for i in range(len(symbols)):
    data_or = pd.read_csv("/Users/wuyong/alldata/original_data/BITFINEX_" + symbols[i] + "_4h_2017-01-01_2018-12-21.csv", index_col=0)
    data_or["date"] = pd.to_datetime(data_or["date"])
    # print(type(data_or["date"].values[0]))
    data_or = data_or.merge(data_result[[symbols[i],"date"]], on="date", how="left")
    data_or.rename(columns={symbols[i]: factor_name}, inplace=True)
    # print(data_or.tail())
    data_or.to_csv("/Users/wuyong/alldata/factor_writedb/factor_stra_4h/"+symbols[i]+"_Alpha.alpha222_gtja4h.csv")
    # break













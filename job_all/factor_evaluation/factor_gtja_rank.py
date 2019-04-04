# coding=utf-8

import sys
sys.path.append('..')
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from lib.myfun import *
import warnings
import copy
warnings.filterwarnings("ignore")

#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)

exchange = 'BITFINEX'

symbols = ["xrpbtc", "ethbtc", "mdabtc", "eosbtc", "xlmbtc", "tusdbtc", "ltcbtc",
           "stratbtc", "trxbtc", "adabtc", "iotabtc", "xmrbtc", "bnbbtc", "dashbtc",
           "xembtc", "etcbtc", "neobtc", "ontbtc", "zecbtc", "wavesbtc", "btgbtc",
           "vetbtc", "qtumbtc", "omgbtc", "zrxbtc", "gvtbtc", "bchabcbtc", "bchsvbtc"]


for i in range(len(symbols)):
    dataf = pd.read_csv("/Users/wuyong/alldata/original_data/BIAN_" + symbols[i] + "_4h_2018-01-01_2019-02-14.csv", index_col=0)
    # print(symbols[i])
    dataf["open_" + symbols[i]] = dataf["open"]
    dataf["close_" + symbols[i]] = dataf["close"]
    dataf["high_" + symbols[i]] = dataf["high"]
    dataf["low_" + symbols[i]] = dataf["low"]
    dataf["volume_" + symbols[i]] = dataf["volume"]
    dataf["amount_" + symbols[i]] = dataf["amount"]

    if i == 0:
        data_open = dataf[["open_"+symbols[i], "date"]]
        data_close = dataf[["close_" + symbols[i], "date"]]
        data_high = dataf[["high_" + symbols[i], "date"]]
        data_low = dataf[["low_" + symbols[i], "date"]]
        data_volume = dataf[["volume_" + symbols[i], "date"]]
        data_amount = dataf[["amount_" + symbols[i], "date"]]

    else:
        data_open = data_open.merge(dataf[["open_"+symbols[i], "date"]], how="left",on="date")
        data_close = data_close.merge(dataf[["close_"+symbols[i], "date"]], how="left",on="date")
        data_high = data_high.merge(dataf[["high_" + symbols[i], "date"]], how="left", on="date")
        data_low = data_low.merge(dataf[["low_" + symbols[i], "date"]], how="left", on="date")
        data_volume = data_volume.merge(dataf[["volume_" + symbols[i], "date"]], how="left", on="date")
        data_amount = data_amount.merge(dataf[["amount_" + symbols[i], "date"]], how="left", on="date")


def rolling_rank(na):
    return rankdata(na)[-1]


def ts_rank(df, window=10):
    return window+1-df.rolling(window).apply(rolling_rank)


def stddev(df, window=10):
    return df.rolling(window).std()


def ts_sum(df, window=10):
    return df.rolling(window).sum()


def ts_max(df ,window=10):
    return df.rolling(window).max()


def ts_mean(df ,window=10):
    return df.rolling(window).mean()


def rolling_prod(na):
    return np.prod(na)


def product(df, window=10):
    return df.rolling(window).apply(rolling_prod)


def decay_linear(df, period=10):
    if df.isnull().values.any():
        df.fillna(method='ffill', inplace=True)
    na_lwma = np.zeros_like(df)
    na_lwma[:period, :] = df.iloc[:period, :]
    na_series = df.as_matrix()

    divisor = period * (period + 1) / 2
    y = (np.arange(period) + 1) * 1.0 / divisor
    for row in range(period - 1, df.shape[0]):
        x = na_series[row - period + 1: row + 1, :]
        na_lwma[row, :] = (np.dot(x.T, y))
    return pd.DataFrame(na_lwma, index=df.index, columns=df.columns)


data_list = [data_open, data_high, data_low, data_close, data_volume, data_amount]


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
        self.amount = pn_data[5]
        del self.amount["date"]
        self.returns = pd.DataFrame((self.close.values/self.close.shift(1).values)-1, columns=self.close.columns)
        self.vwap = pd.DataFrame(self.amount.values/(self.volume.values*100), columns=self.close.columns)

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

    def alpha006(self):
        data_mid1 = pd.DataFrame((self.open.values*0.85+self.high.values*0.15), columns=self.open.columns)
        data_mid1 = data_mid1.diff(4)
        data_mid1 = data_mid1.apply(lambda x: np.sign(x))
        data_mid1 = data_mid1.rank(axis=1,numeric_only=True,na_option="keep")*(-1)
        data_mid1["date"] = self.date
        data_mid1.columns = symbols+["date"]
        return data_mid1

    def alpha007(self):
        # ((RANK(MAX((VWAP - CLOSE), 3)) + RANK(MIN((VWAP - CLOSE), 3))) * RANK(DELTA(VOLUME, 3)))
        data_mid1 = self.volume.diff(3).rank(axis=1, numeric_only=True, na_option="keep")
        data_mid2 = pd.DataFrame(np.maximum(self.vwap.values-self.close.values, 3), columns=self.close.columns).rank(axis=1, numeric_only=True,na_option="keep")
        data_mid3 = pd.DataFrame(np.minimum(self.vwap.values-self.close.values, 3), columns=self.close.columns).rank(axis=1, numeric_only=True,na_option="keep")
        result = pd.DataFrame((data_mid2.values+data_mid3.values)*data_mid1.values, columns=self.close.columns)
        result["date"] = self.date
        result.columns = symbols + ["date"]
        return result

    def alpha008(self):
        data_mid1 = pd.DataFrame((((self.high.values+self.low.values)/2)*0.2)+(self.vwap.values*0.8), columns=self.close.columns)
        data_mid1 = -1*data_mid1.diff(4)
        result = data_mid1.rank(axis=1, numeric_only=True, na_option="keep")
        result["date"] = self.date
        result.columns = symbols + ["date"]
        return result

    def alpha010(self):
        data_mid1 = stddev(self.returns, 20)
        data_mid1[self.returns >= 0] = self.close
        data_mid1 = pd.DataFrame(np.maximum(data_mid1.values**2, 0.0001), columns=self.close.columns).rank(axis=1, numeric_only=True, na_option="keep")
        result = data_mid1
        result["date"] = self.date
        result.columns = symbols + ["date"]
        return result

    def alpha012(self):
        data_mid1 = ts_sum(self.vwap, 10)/10
        data_mid1 = pd.DataFrame(self.open.values-data_mid1.values, columns=self.close.columns).rank(axis=1, numeric_only=True, na_option="keep")
        data_mid2 = pd.DataFrame(self.close.values-self.vwap.values, columns=self.close.columns).abs()
        data_mid2 = -1*data_mid2.rank(axis=1, numeric_only=True, na_option="keep")
        result = data_mid1*data_mid2
        result["date"] = self.date
        result.columns = symbols + ["date"]
        return result

    def alpha016(self):
        data_mid1 = pd.DataFrame(self.volume.values, columns=self.close.columns).rank(axis=1, numeric_only=True, na_option="keep")
        data_mid2 = self.vwap.rank(axis=1, numeric_only=True, na_option="keep")
        data_mid3 = data_mid2.rolling(window=10).corr(data_mid1).rank(axis=1, numeric_only=True, na_option="keep")
        result = ts_max(data_mid3, 10)*-1
        result["date"] = self.date
        result.columns = symbols + ["date"]
        return result

    def alpha017(self):
        data_mid1 = pd.DataFrame(self.vwap.values-np.maximum(self.vwap.values, 15), columns=self.close.columns).rank(axis=1, numeric_only=True, na_option="keep")
        data_mid2 = self.close.diff(5)
        result = data_mid1**data_mid2
        result["date"] = self.date
        result.columns = symbols + ["date"]
        return result

    def alpha025(self):
        data_mid1 = self.close.diff(7)
        data_mid2 = self.volume/ts_mean(self.volume, 20)
        data_mid2 = 1-decay_linear(data_mid2, 9).rank(axis=1, numeric_only=True, na_option="keep")
        data_mid3 = pd.DataFrame(data_mid1.values*data_mid2.values, columns=self.close.columns).rank(axis=1, numeric_only=True, na_option="keep")*-1
        data_mid4 = ts_sum(self.returns,250).rank(axis=1, numeric_only=True, na_option="keep")+1
        result = pd.DataFrame(data_mid3.values*data_mid4.values, columns=self.close.columns)
        result["date"] = self.date
        result.columns = symbols + ["date"]
        return result

    def alpha0032(self):
        data_mid1 = self.high.rank(axis=1,numeric_only=True,na_option="keep")
        data_mid2 = pd.DataFrame(self.volume.values, columns=self.high.columns).rank(axis=1, numeric_only=True, na_option="keep")
        result = data_mid2.rolling(window=10).corr(data_mid1)
        result = result.rank(axis=1,numeric_only=True,na_option="keep").rolling(window=10).sum()*-1
        result["date"] = self.date
        result.columns = symbols+["date"]
        return result

    def alpha033(self):
        data_mid1 = ts_rank(self.volume,window=5)
        data_mid2 = -1*(self.low.rolling(window=5).min())+(self.low.rolling(window=5).min()).shift(5)
        data_mid3 = ((self.returns.rolling(window=240).sum()-self.returns.rolling(window=20).sum())/220).rank(axis=1,numeric_only=True,na_option="keep")
        result = pd.DataFrame(data_mid1.values*data_mid2.values*data_mid3.values, columns=self.close.columns)
        result["date"] = self.date
        result.columns = symbols+["date"]
        return result

    def alpha035(self):
        data_mid1 = decay_linear(self.open.diff(), 15).rank(axis=1,numeric_only=True,na_option="keep")
        data_mid2 = pd.DataFrame(self.volume.values, columns=self.open.columns).rolling(window=17).corr(self.open)
        data_mid2 = decay_linear(data_mid2, 7).rank(axis=1, numeric_only=True, na_option="keep")
        result = pd.DataFrame(np.minimum(data_mid1.values, data_mid2.values), columns=self.close.columns)*-1
        result["date"] = self.date
        result.columns = symbols + ["date"]
        return result

    def alpha036(self):
        data_mid1 = self.volume.rank(axis=1,numeric_only=True,na_option="keep")
        data_mid2 = self.vwap.rank(axis=1,numeric_only=True,na_option="keep")
        data_mid3 = pd.DataFrame(data_mid1.values, columns=self.vwap.columns).rolling(window=6).corr(data_mid2)
        result = ts_sum(data_mid3, 2).rank(axis=1,numeric_only=True,na_option="keep")
        result["date"] = self.date
        result.columns = symbols + ["date"]
        return result

    def alpha037(self):
        data_mid1 = self.open.rolling(window=5).sum()
        data_mid2 = self.returns.rolling(window=5).sum()
        data_mid3 = pd.DataFrame(data_mid1.values*data_mid2.values, columns=self.close.columns)
        data_mid3 = (data_mid3-data_mid3.shift(10)).rank(axis=1,numeric_only=True,na_option="keep")*-1
        data_mid3["date"] = self.date
        data_mid3.columns = symbols+["date"]
        return data_mid3

    def alpha039(self):
        data_mid1 = decay_linear(self.close.diff(2), 8).rank(axis=1, numeric_only=True, na_option="keep")
        data_mid2 = pd.DataFrame(self.vwap.values*0.3+self.open.values*0.7, columns=self.close.columns)
        data_mid3 = ts_sum(ts_mean(self.volume,180), 37)
        data_mid3 = pd.DataFrame(data_mid3.values, columns=self.close.columns)
        data_mid4 = data_mid2.rolling(window=14).corr(data_mid3)
        data_mid4 = decay_linear(data_mid4, 12).rank(axis=1,numeric_only=True,na_option="keep")
        result = (data_mid1-data_mid4)*-1
        result["date"] = self.date
        result.columns = symbols + ["date"]
        return result

    def alpha041(self):
        data_mid1 = pd.DataFrame(np.maximum(self.vwap.diff(3), 0), columns=self.close.columns)
        result = data_mid1.rank(axis=1, numeric_only=True, na_option="keep")*-1
        result["date"] = self.date
        result.columns = symbols + ["date"]
        return result

    def alpha042(self):
        data_mid1 = self.high.rolling(window=10).std().rank(axis=1,numeric_only=True,na_option="keep")*-1
        data_mid2 = pd.DataFrame(self.volume.values, columns=self.high.columns)
        data_mid2 = self.high.rolling(window=10).corr(data_mid2)
        result = data_mid1*data_mid2
        result["date"] = self.date
        result.columns = symbols+["date"]
        return result

    def alpha044(self):
        data_mid1 = pd.DataFrame(self.low.values, columns=self.volume.columns).rolling(window=7).corr(ts_mean(self.volume, 10))
        data_mid1 = decay_linear(data_mid1, 6)
        data_mid1 = ts_rank(data_mid1, 4)
        data_mid2 = decay_linear(self.vwap.diff(3), 10)
        data_mid2 = ts_rank(data_mid2, 15)
        data_mid2.columns = data_mid1.columns
        result = data_mid1+data_mid2
        result["date"] = self.date
        result.columns = symbols + ["date"]
        return result

    def alpha045(self):
        data_mid1 = pd.DataFrame(self.close.values*0.6+self.open.values*0.4, columns=self.close.columns).diff()
        data_mid1 = data_mid1.rank(axis=1,numeric_only=True,na_option="keep")
        data_mid2 = pd.DataFrame(self.vwap.values, columns=self.volume.columns).rolling(window=15).corr(ts_mean(self.volume, 150))
        data_mid2 = data_mid2.rank(axis=1,numeric_only=True,na_option="keep")
        result = pd.DataFrame(data_mid1.values*data_mid2.values, columns=self.close.columns)
        result["date"] = self.date
        result.columns = symbols + ["date"]
        return result

    def alpha048(self):
        data_mid1 = (self.close-self.close.shift(1)).apply(lambda x: np.sign(x))+\
                    (self.close.shift(1)-self.close.shift(2)).apply(lambda x: np.sign(x))+\
                    (self.close.shift(2)-self.close.shift(3)).apply(lambda x: np.sign(x))
        data_mid1 = data_mid1.rank(axis=1, numeric_only=True,na_option="keep")
        data_mid2 = pd.DataFrame(self.volume.values, columns=self.close.columns)
        data_mid2 = data_mid2.rolling(window=5).sum()*data_mid1
        data_mid3 = pd.DataFrame(self.volume.values, columns=self.close.columns).rolling(window=20).sum()
        result = -1*data_mid2/data_mid3
        result["date"] = self.date
        result.columns = symbols+["date"]
        return result

    def alpha054(self):
        data_mid1 = pd.DataFrame(self.open.values, columns=self.close.columns)
        data_mid2 = ((self.close-data_mid1).apply(lambda x: np.abs(x))).rolling(window=10).std()
        data_mid3 = self.close-data_mid1
        data_mid4 = self.close.rolling(window=10).corr(data_mid1)
        result = -1*(data_mid2+data_mid3+data_mid4).rank(axis=1,numeric_only=True,na_option="keep")
        result["date"] = self.date
        result.columns = symbols+["date"]
        return result

    def alpha061(self):
        data_mid1 = pd.DataFrame(self.low.values, columns=self.volume.columns).rolling(window=8).corr(ts_mean(self.volume, 80))
        data_mid1 = data_mid1.rank(axis=1,numeric_only=True,na_option="keep")
        data_mid1 = decay_linear(data_mid1, 17).rank(axis=1,numeric_only=True,na_option="keep")
        data_mid2 = decay_linear(self.vwap.diff(), 12).rank(axis=1,numeric_only=True,na_option="keep")
        result = pd.DataFrame(np.maximum(data_mid1.values, data_mid2.values), columns=self.close.columns)*-1
        result["date"] = self.date
        result.columns = symbols + ["date"]
        return result

    def alpha062(self):
        data_mid1 = self.volume.rank(axis=1,numeric_only=True,na_option="keep")
        data_mid1 = pd.DataFrame(data_mid1.values, columns=self.high.columns)
        result = -1*(data_mid1.rolling(window=10).corr(self.high))
        result["date"] = self.date
        result.columns = symbols+["date"]
        return result

    # def alpha064(self):
    #     data_mid1 = self.volume.rank(axis=1, numeric_only=True, na_option="keep")
    #     data_mid2 = self.vwap.rank(axis=1, numeric_only=True, na_option="keep")
    #     data_mid3 = pd.DataFrame(data_mid1.values, columns=self.vwap.columns).rolling(window=4).corr(data_mid2)
    #     data_mid3 = decay_linear(data_mid3, 4).rank(axis=1,numeric_only=True,na_option="keep")
    #     data_mid1 = self.close.rank(axis=1, numeric_only=True, na_option="keep")
    #     data_mid2 = ts_mean(self.volume, 60).rank(axis=1, numeric_only=True, na_option="keep")
    #     data_mid4 = pd.DataFrame(data_mid2.values, columns=self.close.columns).rolling(window=4).corr(data_mid1)
    #     data_mid4 = pd.DataFrame(np.maximum())  # 这个地方的公式是有问题的
    #     pass

    def alpha073(self):
        data_mid1 = pd.DataFrame(self.volume.values, columns=self.close.columns).rolling(window=10).corr(self.close)
        data_mid1 = decay_linear(decay_linear(data_mid1, 16), 4)
        data_mid1 = ts_rank(data_mid1, 5)
        data_mid2 = pd.DataFrame(self.vwap.values, columns=self.volume.columns).rolling(window=4).corr(ts_mean(self.volume, 30))
        data_mid2 = decay_linear(data_mid2, 3).rank(axis=1,numeric_only=True,na_option="keep")
        data_mid2 = pd.DataFrame(data_mid2.values, columns=self.close.columns)
        result = (data_mid1-data_mid2)*-1
        result["date"] = self.date
        result.columns = symbols + ["date"]
        return result

    def alpha074(self):
        data_mid1 = pd.DataFrame(self.low.values+self.vwap.values, columns=self.volume.columns)
        data_mid1 = ts_sum(data_mid1, 20)
        data_mid2 = ts_sum(ts_mean(self.volume, 40), 20)
        data_mid3 = (data_mid1.rolling(window=7).corr(data_mid2)).rank(axis=1,numeric_only=True,na_option="keep")
        data_mid4 = pd.DataFrame(self.vwap.values, columns=self.volume.columns).rank(axis=1,numeric_only=True,na_option="keep")
        data_mid4 = (data_mid4.rolling(window=6).corr(self.volume.rank(axis=1,numeric_only=True,na_option="keep"))).rank(axis=1,numeric_only=True,na_option="keep")
        result = data_mid3+data_mid4
        result["date"] = self.date
        result.columns = symbols + ["date"]
        return result

    def alpha077(self):
        data_mid1 = pd.DataFrame((self.high.values+self.low.values)/2-self.vwap.values, columns=self.volume.columns)
        data_mid1 = decay_linear(data_mid1, 20).rank(axis=1,numeric_only=True,na_option="keep")
        data_mid2 = pd.DataFrame((self.high.values+self.low.values)/2, columns=self.volume.columns).rolling(window=3).corr(ts_mean(self.volume, 40))
        data_mid2 = decay_linear(data_mid2, 6).rank(axis=1,numeric_only=True,na_option="keep")
        result = pd.DataFrame(np.minimum(data_mid1.values, data_mid2.values), columns=self.close.columns)
        result["date"] = self.date
        result.columns = symbols + ["date"]
        return result

    def alpha083(self):
        data_mid1 = pd.DataFrame(self.volume.rank(axis=1,numeric_only=True,na_option="keep").values, columns=self.high.columns)
        data_mid2 = self.high.rank(axis=1,numeric_only=True,na_option="keep")
        result = data_mid1.rolling(window=5).cov(data_mid2).rank(axis=1,numeric_only=True,na_option="keep")*-1
        result["date"] = self.date
        result.columns = symbols+["date"]
        return result

    def alpha087(self):
        data_mid1 = decay_linear(self.vwap.diff(4), 7).rank(axis=1,numeric_only=True,na_option="keep")
        data_mid2 = pd.DataFrame((self.low.values-self.vwap.values)/(self.open.values-(self.high.values+self.low.values)/2), columns=self.vwap.columns)
        data_mid2 = decay_linear(data_mid2, 11)
        data_mid2 = ts_rank(data_mid2, 7)
        result = (data_mid1+data_mid2)*-1
        result["date"] = self.date
        result.columns = symbols + ["date"]
        return result

    def alpha090(self):
        data_mid1 = self.volume.rank(axis=1, numeric_only=True, na_option="keep")
        data_mid2 = self.vwap.rank(axis=1, numeric_only=True, na_option="keep")
        data_mid3 = pd.DataFrame(data_mid1.values, columns=self.vwap.columns).rolling(window=5).corr(data_mid2)
        result = data_mid3.rank(axis=1,numeric_only=True,na_option="keep")*-1
        result["date"] = self.date
        result.columns = symbols + ["date"]
        return result

    def alpha091(self):
        data_mid1 = (self.close-self.close.rolling(window=5).max()).rank(axis=1,numeric_only=True,na_option="keep")
        data_mid2 = pd.DataFrame(self.volume.rolling(window=40).mean().values, columns=self.low.columns)
        data_mid2 = data_mid2.rolling(window=5).corr(self.low).rank(axis=1,numeric_only=True,na_option="keep")
        result = data_mid1*pd.DataFrame(data_mid2.values, columns=self.close.columns)*-1
        result["date"] = self.date
        result.columns = symbols+["date"]
        return result

    def alpha092(self):
        data_mid1 = pd.DataFrame(self.close.values*0.35+self.vwap.values*0.95, columns=self.close.columns)
        data_mid1 = decay_linear(data_mid1.diff(2), 3).rank(axis=1,numeric_only=True,na_option="keep")
        data_mid2 = (pd.DataFrame(ts_mean(self.volume, 180).values, columns=self.close.columns).rolling(window=13).corr(self.close)).abs()
        data_mid2 = ts_rank(decay_linear(data_mid2, 5), 15)
        result = pd.DataFrame(np.maximum(data_mid1.values, data_mid2.values), columns=self.close.columns)*-1
        result["date"] = self.date
        result.columns = symbols + ["date"]
        return result

    def alpha099(self):
        data_mid1 = pd.DataFrame(self.volume.values, columns=self.close.columns).rank(axis=1,numeric_only=True,na_option="keep")
        data_mid2 = ((self.close.rank(axis=1,numeric_only=True,na_option="keep")).rolling(window=5).cov(data_mid1)).rank(axis=1,numeric_only=True,na_option="keep")*-1
        data_mid2["date"] = self.date
        data_mid2.columns = symbols+["date"]
        return data_mid2

    def alpha104(self):
        data_mid1 = self.high.rolling(window=5).corr(pd.DataFrame(self.volume.values, columns=self.high.columns))
        data_mid1 = data_mid1-data_mid1.shift(5)
        data_mid2 = (self.close.rolling(window=20).std()).rank(axis=1,numeric_only=True,na_option="keep")
        result = -1*(pd.DataFrame(data_mid1.values*data_mid2.values, columns=self.close.columns))
        result["date"] = self.date
        result.columns = symbols+["date"]
        return result

    def alpha105(self):
        data_mid1 = self.open.rank(axis=1,numeric_only=True,na_option="keep")
        data_mid2 = (pd.DataFrame(self.volume.values, columns=self.open.columns)).rank(axis=1,numeric_only=True,na_option="keep")
        result = data_mid1.rolling(window=10).corr(data_mid2)*-1
        result["date"] = self.date
        result.columns = symbols+["date"]
        return result

    def alpha107(self):
        data_mid1 = (self.open-pd.DataFrame(self.high.values, columns=self.open.columns).shift(1)).rank(axis=1, numeric_only=True,na_option="keep")
        data_mid2 = (self.open-pd.DataFrame(self.close.values, columns=self.open.columns).shift(1)).rank(axis=1, numeric_only=True,na_option="keep")
        data_mid3 = (self.open-pd.DataFrame(self.low.values, columns=self.open.columns).shift(1)).rank(axis=1, numeric_only=True,na_option="keep")
        result = -1*data_mid1*data_mid2*data_mid3
        result["date"] = self.date
        result.columns = symbols+["date"]
        return result

    def alpha108(self):
        data_mid1 = pd.DataFrame(self.high.values-np.minimum(self.high.values, 2), columns=self.close.columns).rank(axis=1, numeric_only=True,na_option="keep")
        data_mid2 = pd.DataFrame(self.vwap.values, columns=self.volume.columns).rolling(window=6).corr(
            ts_mean(self.volume, 120))
        data_mid2 = data_mid2.rank(axis=1, numeric_only=True, na_option="keep")
        result = pd.DataFrame(data_mid1.values**data_mid2.values, columns=self.close.columns)*-1
        result["date"] = self.date
        result.columns = symbols + ["date"]
        return result

    def alpha119(self):
        data_mid4 = pd.DataFrame(ts_sum(ts_mean(self.volume, 5), 26), columns=self.vwap.columns).rolling(window=5).corr(self.vwap)
        data_mid4 = decay_linear(data_mid4, 7).rank(axis=1, numeric_only=True, na_option="keep")
        data_mid1 = self.open.rank(axis=1, numeric_only=True, na_option="keep")
        data_mid2 = pd.DataFrame(ts_mean(self.volume, 15), columns=self.open.columns).rank(axis=1, numeric_only=True, na_option="keep")
        data_mid3 = data_mid1.rolling(window=21).corr(data_mid2)
        data_mid3 = decay_linear(ts_rank(data_mid3, 7), 8).rank(axis=1, numeric_only=True, na_option="keep")
        # result = data_mid4-data_mid3
        result = pd.DataFrame(data_mid4.values-data_mid3.values, columns=self.close.columns)
        result["date"] = self.date
        result.columns = symbols + ["date"]
        return result

    def alpha120(self):
        data_mid1 = pd.DataFrame(self.vwap.values-self.close.values, columns=self.close.columns).rank(axis=1, numeric_only=True, na_option="keep")
        data_mid2 = pd.DataFrame(self.vwap.values+self.close.values, columns=self.close.columns).rank(axis=1, numeric_only=True, na_option="keep")
        result = pd.DataFrame(data_mid1.values/data_mid2.values, columns=self.close.columns)
        result["date"] = self.date
        result.columns = symbols + ["date"]
        return result

    # def alpha121(self):
    #     data_mid1 = pd.DataFrame(self.vwap.values-np.minimum(self.vwap.values, 12))
    #     pass

    def alpha124(self):
        data_mid1 = pd.DataFrame(self.close.values-self.vwap.values, columns=self.close.columns)
        data_mid2 = ts_max(self.close, 30).rank(axis=1, numeric_only=True, na_option="keep")
        data_mid2 = decay_linear(data_mid2, 2)
        result = data_mid1/data_mid2
        result["date"] = self.date
        result.columns = symbols + ["date"]
        return result

    def alpha125(self):
        data_mid1 = pd.DataFrame(self.vwap.values, columns=self.volume.columns).rolling(window=17).corr(ts_mean(self.volume, 80))
        data_mid1 = decay_linear(data_mid1, 20).rank(axis=1, numeric_only=True, na_option="keep")
        data_mid2 = pd.DataFrame(self.close.values*0.5+self.vwap.values*0.5, columns=self.volume.columns).diff(3)
        data_mid2 = decay_linear(data_mid2, 16).rank(axis=1, numeric_only=True, na_option="keep")
        result = data_mid1/data_mid2
        result["date"] = self.date
        result.columns = symbols + ["date"]
        return result

    def alpha130(self):
        data_mid4 = pd.DataFrame((self.high.values+self.low.values)/2, columns=self.volume.columns).rolling(window=9).corr(ts_mean(self.volume, 40))
        data_mid4 = decay_linear(data_mid4, 10).rank(axis=1, numeric_only=True, na_option="keep")
        data_mid1 = self.volume.rank(axis=1, numeric_only=True, na_option="keep")
        data_mid2 = self.vwap.rank(axis=1, numeric_only=True, na_option="keep")
        data_mid3 = pd.DataFrame(data_mid2.values, columns=self.volume.columns).rolling(window=7).corr(data_mid1)
        data_mid3 = decay_linear(data_mid3, 3).rank(axis=1, numeric_only=True, na_option="keep")
        result = data_mid4/data_mid3
        result["date"] = self.date
        result.columns = symbols + ["date"]
        return result

    def alpha131(self):
        data_mid1 = self.vwap.diff().rank(axis=1, numeric_only=True, na_option="keep")
        data_mid2 = pd.DataFrame(self.close.values, columns=self.volume.columns).rolling(window=18).corr(ts_mean(self.volume, 50))
        data_mid2 = ts_rank(data_mid2, 18)
        result = pd.DataFrame(data_mid1.values/data_mid2.values, columns=self.close.columns)
        result["date"] = self.date
        result.columns = symbols + ["date"]
        return result

    def alpha136(self):
        data_mid1 = self.open.rolling(window=10).corr(pd.DataFrame(self.volume.values, columns=self.open.columns))
        data_mid2 = -1*(self.returns - self.returns.shift(3)).rank(axis=1,numeric_only=True,na_option="keep")
        result = pd.DataFrame(data_mid1.values*data_mid2.values, columns=self.close.columns)
        result["date"] = self.date
        result.columns = symbols+["date"]
        return result

    def alpha138(self):
        data_mid1 = pd.DataFrame(self.low.values*0.7+self.vwap.values*0.3, columns=self.volume.columns).diff(3)
        data_mid1 = decay_linear(data_mid1, 20).rank(axis=1, numeric_only=True, na_option="keep")
        data_mid2 = pd.DataFrame(ts_rank(self.low, 8).values, columns=self.volume.columns).rolling(window=5).corr(ts_rank(ts_mean(self.volume, 60), 17))
        data_mid2 = ts_rank(decay_linear(ts_rank(data_mid2, 19), 16), 7)
        result = (data_mid1-data_mid2)*-1
        result["date"] = self.date
        result.columns = symbols + ["date"]
        return result

    def alpha140(self):
        data_mid1 = self.open.rank(axis=1, numeric_only=True, na_option="keep")
        data_mid2 = self.close.rank(axis=1, numeric_only=True, na_option="keep")
        data_mid3 = self.high.rank(axis=1, numeric_only=True, na_option="keep")
        data_mid4 = self.low.rank(axis=1, numeric_only=True, na_option="keep")
        data_mid5 = pd.DataFrame((data_mid1.values+data_mid4.values)-(data_mid2.values+data_mid3.values), columns=self.close.columns)
        data_mid5 = decay_linear(data_mid5, 8).rank(axis=1, numeric_only=True, na_option="keep")
        data_mid6 = pd.DataFrame(ts_rank(self.close, 8).values, columns=self.volume.columns).rolling(window=8).corr(ts_rank(ts_mean(self.volume, 60), 20))
        data_mid6 = ts_rank(decay_linear(data_mid6, 7), 3)
        result = pd.DataFrame(np.minimum(data_mid5.values, data_mid6.values), columns=self.close.columns)
        result["date"] = self.date
        result.columns = symbols + ["date"]
        return result

    def alpha141(self):
        data_mid1 = (self.volume.rolling(window=15).mean()).rank(axis=1,numeric_only=True,na_option="keep")
        data_mid2 = self.high.rank(axis=1,numeric_only=True,na_option="keep")
        data_mid3 = data_mid2.rolling(window=9).corr(pd.DataFrame(data_mid1.values, columns=self.high.columns))
        result = data_mid3.rank(axis=1,numeric_only=True,na_option="keep")*-1
        result["date"] = self.date
        result.columns = symbols+["date"]
        return result

    def alpha142(self):
        data_mid1 = (ts_rank(self.close, window=10)).rank(axis=1,numeric_only=True,na_option="keep")*-1
        data_mid2 = ((self.close.diff()).diff()).rank(axis=1,numeric_only=True,na_option="keep")
        data_mid3 = (ts_rank(self.volume/(self.volume.rolling(window=20).mean()), window=5)).rank(axis=1,numeric_only=True,na_option="keep")
        result = pd.DataFrame(data_mid1.values*data_mid2.values*data_mid3.values, columns=self.close.columns)
        result["date"] = self.date
        result.columns = symbols+["date"]
        return result

    def alpha156(self):
        data_mid1 = decay_linear(self.vwap.diff(5), 3).rank(axis=1,numeric_only=True,na_option="keep")
        data_mid2 = pd.DataFrame(self.open.values*0.15+self.low.values*0.85, columns=self.close.columns)
        data_mid3 = data_mid2.diff(2)/data_mid2*-1
        data_mid3 = decay_linear(data_mid3, 3).rank(axis=1,numeric_only=True,na_option="keep")
        result = pd.DataFrame(np.maximum(data_mid1.values, data_mid3.values), columns=self.close.columns)*-1
        result["date"] = self.date
        result.columns = symbols + ["date"]
        return result

    def alpha163(self):
        data_mid1 = pd.DataFrame(-1*self.returns.values*ts_mean(self.volume, 20).values*self.vwap.values*(self.high.values-self.close.values), columns=self.close.columns)
        result = data_mid1.rank(axis=1, numeric_only=True, na_option="keep")
        result["date"] = self.date
        result.columns = symbols + ["date"]
        return result

    def alpha170(self):
        data_mid1 = (1/self.close).rank(axis=1,numeric_only=True,na_option="keep")
        data_mid1 = pd.DataFrame(data_mid1.values*self.volume.values/ts_mean(self.volume, 20).values, columns=self.close.columns)
        data_mid2 = pd.DataFrame(self.high.values-self.close.values, columns=self.close.columns).rank(axis=1, numeric_only=True, na_option="keep")
        data_mid2 = pd.DataFrame(self.high.values*data_mid2.values/(ts_sum(self.high, 5).values/5))
        data_mid3 = (self.vwap-self.vwap.shift(5)).rank(axis=1,numeric_only=True,na_option="keep")
        result = pd.DataFrame(data_mid1.values*data_mid2.values-data_mid3.values, columns=self.close.columns)
        result["date"] = self.date
        result.columns = symbols + ["date"]
        return result

    def alpha176(self):
        data_mid1 = self.close-pd.DataFrame(self.low.values, columns=self.close.columns).rolling(window=12).min()
        data_mid2 = (pd.DataFrame(self.high.values, columns=self.close.columns)).rolling(window=12).max()-(pd.DataFrame(self.low.values, columns=self.close.columns)).rolling(window=12).min()
        data_mid3 = (data_mid1/data_mid2).rank(axis=1,numeric_only=True,na_option="keep")
        data_mid4 = (pd.DataFrame(self.volume.values, columns=self.close.columns)).rank(axis=1,numeric_only=True,na_option="keep")
        result = data_mid3.rolling(window=6).corr(data_mid4)
        result["date"] = self.date
        result.columns = symbols+["date"]
        return result

    def alpha179(self):
        data_mid1 = pd.DataFrame(self.vwap.values, columns=self.volume.columns).rolling(window=4).corr(self.volume).rank(axis=1,numeric_only=True,na_option="keep")
        data_mid2 = self.low.rank(axis=1,numeric_only=True,na_option="keep")
        data_mid3 = ts_mean(self.volume, 50).rank(axis=1,numeric_only=True,na_option="keep")
        data_mid4 = pd.DataFrame(data_mid2.values, columns=self.volume.columns).rolling(window=12).corr(data_mid3).rank(axis=1,numeric_only=True,na_option="keep")
        result = data_mid1*data_mid4
        result["date"] = self.date
        result.columns = symbols + ["date"]
        return result

    def alpha184(self):
        data_mid1 = (pd.DataFrame(self.open.values, columns=self.close.columns)-self.close).shift(1)
        data_mid1 = (data_mid1.rolling(window=200).corr(self.close)).rank(axis=1,numeric_only=True,na_option="keep")
        data_mid2 = (pd.DataFrame(self.open.values, columns=self.close.columns)-self.close).rank(axis=1,numeric_only=True,na_option="keep")
        result = data_mid1+data_mid2
        result["date"] = self.date
        result.columns = symbols+["date"]
        return result

    def alpha185(self):
        data_mid1 = 1-(pd.DataFrame(self.open.values, columns=self.close.columns)/self.close)
        data_mid1 = (data_mid1**2)*-1
        result = data_mid1.rank(axis=1, numeric_only=True, na_option="keep")
        result["date"] = self.date
        result.columns = symbols+["date"]
        return result


Alpha = Alphas(data_list)
# print(Alpha.alpha021().head(30))
# print(data_result.head(50))
# print(data_result.tail())
# print(type(data_result["date"].values[0]))


"""
下面是一次性计算出各个币对所有101因子的因子值，然后将单个因子值数据和单个币对的行情数据放在一个DataFrame当中存到本地备用
"""

a = list(range(1,202))
alpha_test = []
for x in a:
    if x < 10:
        alpha_test.append("Alpha.alpha00"+str(x))
    elif 10 < x < 100:
        alpha_test.append("Alpha.alpha0"+str(x))
    else:
        alpha_test.append("Alpha.alpha" + str(x))

# 对alpha_test当中的因子循环计算出所有币对的因子值
for factor in alpha_test:
    try:
        data_result = eval(factor)()  # 一次性计算出某个因子所有28个币对的因子值数据
        for symbol in symbols:
            dataf = pd.read_csv("/Users/wuyong/alldata/original_data/BIAN_"+symbol+"_4h_2018-01-01_2019-02-14.csv", index_col=0)  # 这个地方调用某个币对的行情数据
            col_name = factor
            df_m = copy.deepcopy(dataf)
            data_factor = data_result[[symbol, "date"]]
            df_m = df_m.merge(data_factor, on="date", how="left")  # 这个地方把币对的行情数据和该币对的某个因子值数据结合在一起
            df_m[col_name] = df_m[symbol].values
            del df_m[symbol]
            factor_name = factor + "_" + "gtja4h"
            fname = '/Users/wuyong/alldata/factor_writedb/factor_stra_4h/BIAN_' + symbol + "_" + factor_name + '.csv'  # 结合之后的数据的存储地址，文件名与币对名和因子名有关
            write_db(df_m, fname, False)
            print('write' + fname + '...')
    except (AttributeError, FileNotFoundError):
        print(factor)










import sys
sys.path.append('..')

import pandas as pd
from lib.myfun import *
import numpy as np
import time
import talib as ta
from datetime import datetime
import matplotlib.pyplot as plt
# 显示所有行
pd.set_option('display.max_rows', None)
np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)


def timestamp_to_datetime(timestamp):
    """将 13 位整数的毫秒时间戳转化成本地普通时间格式)
    :param sp: 13 位整数的毫秒时间戳1456402864242)
    :return: 返回格式 {}2016-02-25 20:21:04.242000
    """
    local_dt_time = datetime.fromtimestamp(timestamp)
    return local_dt_time


# data_results = pd.read_csv("/Users/wuyong/alldata/original_data/trades_BIAN_btcusdt_all.csv")
# data_results["dealtime"] = data_results["dealtime"].apply(lambda x: int(x/1000))
# data_results.drop_duplicates(keep="last",inplace=True,subset="dealtime")
# data_last = pd.DataFrame({"dealtime":range(1502942429,1545237977)})
# print(data_results.head())
# print(data_results.tail())
# print(len(data_last),len(data_results))
# data_last = data_last.merge(data_results,how="left",on="dealtime")
# data_last.fillna(method="ffill",inplace=True)
# print(data_last.head(100))
# data_last.to_csv("/Users/wuyong/alldata/original_data/trades_BIAN_ethusdt_s_all.csv")


# symbol = "xrpusdt"
#
# exchange = 'BIAN'
#
# dataf = read_data(exchange, symbol, '1m', "2017-08-16", "2018-12-21")
# dataf["tickid"] = dataf["tickid"]+60
# dataf = dataf.iloc[2:]
# dataf["date_time"] = pd.to_datetime(dataf["date"])
# df = dataf.resample(rule="3min", on='date_time',label="left", closed="left").apply(
#                 {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
#                  'volume': 'sum', "amount": "sum", "tickid": "first"})
# print(dataf.head(10))
# print(dataf.tail())
# # print(dataf[dataf["tickid"]>1542160420].head(70))
# df["tickid"] = df["tickid"]+180
# df.index = range(len(df))
# # df.fillna(method="ffill",inplace=True)
# # print(df[df["tickid"]>1542935686].head(10))
# # df["tickid_str"] = df["tickid"].apply(lambda x: str(x))
# df["ma90"] = ta.MA(df["close"].values, timeperiod=90, matype=0)
#
# df_all = pd.DataFrame({"tickid": range(1541693160, 1544457660, 180)})
# df_all = df_all.merge(df, how="left", on="tickid")
# df_all.fillna(method="ffill", inplace=True)
#
#
# dataf["ma90"] = ta.MA(dataf["close"].values, timeperiod=90, matype=0)
# print(dataf.tail())
# # print(df_all[["close","ma90","tickid"]])
#
#
# # print(df.loc["2018-12-05 10:36:00"])
# dataf.to_csv("/Users/wuyong/alldata/original_data/trades_bian_xrpusdt_m_all.csv")
#
#
# # data_test = pd.read_csv("/Users/wuyong/alldata/original_data/trades_bian_xrpusdt_5m_2.csv").tail(100)
# #
# # data_test["ma90"] = ta.MA(data_test["close"].values, timeperiod=90, matype=0)
# # print(data_test.head(10))
# # print(data_test.tail())

# data_test = pd.read_csv("/Users/wuyong/alldata/original_data/eos_asset.csv",index_col=0)
#
# data_test["date_time"] = [timestamp_to_datetime(x-60) for x in data_test["tickid"].values]
#
# data_test = data_test.resample(rule="1d", on="date_time", label="left", closed="left").apply({"close": "last", "asset": "last"})
# print(data_test.head())
# print(data_test.tail())
# data_test["close"] = data_test["close"]/data_test["close"].values[0]
# data_test["asset"] = data_test["asset"]/data_test["asset"].values[0]
# data_test.plot()
# plt.show()


# data_test = pd.read_csv("/Users/wuyong/alldata/original_data/BITMEX_.bxbt_4h_2018-06-20_2018-12-26.csv", index_col=0)
# data_test = data_test.iloc[19:]
# data_test["date_time"] = pd.to_datetime(data_test["date"])
# data_test = data_test.resample(rule="4h", on="date_time",label="left", closed="left").apply({'close': 'last'})
# data_test.to_csv("/Users/wuyong/alldata/original_data/BITMEX_.bxbt_4h_2018-06-20_2018-12-26.csv")
# print(data_test.head())
# print(len(data_test.iloc[:1126]))


data_3mon_all = pd.read_csv("/Users/wuyong/alldata/original_data/trades_BIAN_btcusdt_s_all.csv", index_col=0)
data_3mon_all["date"] = pd.to_datetime(data_3mon_all["dealtime"], unit="s")
data_3mon_all.index = data_3mon_all["date"].values
print(data_3mon_all.head(40))
data_3mon_all_re = data_3mon_all.resample(rule="30s", on='date', label="left", closed="left").apply(
                {"price": "last", "amount": "sum", "date": "first", "dealtime": "first"})
data_3mon_all_re["close"] = data_3mon_all_re["price"]
data_3mon_all_re["open"] = data_3mon_all.resample(rule="30s", on='date', label="left", closed="left").apply(
                {"price": "first"})
data_3mon_all_re["high"] = data_3mon_all.resample(rule="30s", on='date', label="left", closed="left").apply(
                {"price": "max"})
data_3mon_all_re["low"] = data_3mon_all.resample(rule="30s", on='date', label="left", closed="left").apply(
                {"price": "min"})

data_3mon_all_re["tickid"] = data_3mon_all_re["dealtime"]
data_3mon_all_re["tickid"] = data_3mon_all_re["tickid"]+30
data_3mon_all_re.index = range(len(data_3mon_all_re))
data_3mon_all_re.to_csv("/Users/wuyong/alldata/original_data/trades_BIAN_btcusdt_30s_all.csv")
print(data_3mon_all_re.head(10))













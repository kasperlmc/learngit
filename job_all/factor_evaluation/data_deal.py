import sys
sys.path.append('..')

import pandas as pd
from lib.myfun import *
import numpy as np
import time
from datetime import datetime
# 显示所有行
pd.set_option('display.max_rows', None)

symbol = "btcusdt"

exchange = 'BITFINEX'

dataf = read_data(exchange, symbol, '1h', "2017-01-01", "2018-12-21")
print(dataf.head())
print(dataf.tail())
dataf["date_time"] = pd.to_datetime(dataf["date"])
df = dataf.resample(rule="4h", on='date_time',label="left").apply(
                {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
                 'volume': 'sum', "amount": "sum"})
df["date"] = df.index
df.index = range(len(df))
print(df.head())
print(df.tail())
# df.to_csv("/Users/wuyong/alldata/original_data/BITMEX_.bxbt_4h_2017-01-01_2018-12-21.csv")
# above = dataf.iloc[:350]
# # print(above.tail())
# below = dataf.iloc[350:]
#
# insertRow = dataf.iloc[[349]]
# insertRow.loc[349, "date"] = "2017-01-15 17:00:00"
# insertRow.loc[349,["open","high","low"]] = 9.7147
#
# newData2 = pd.concat([above,insertRow,below],ignore_index=True)
# # print(newData2.head(400))
#
# above = newData2.iloc[:852]
# # print(above.tail())
# below = newData2.iloc[852:]
#
#
# insertRow = newData2.iloc[[851]]
# insertRow.loc[851, "date"] = "2017-02-05 23:00:00"
# insertRow.loc[851,["open", "high", "low"]] = 11.221
#
# newData2 = pd.concat([above,insertRow,below],ignore_index=True)
# print(len(newData2))
# print(newData2.tail())
#
# newData2.to_csv("/Users/wuyong/alldata/original_data/BITFINEX_ethusdt_1h_2017-01-01_2018-10-01.csv")


# # symbol = ".bxbt"
#
# exchange = 'BITMEX'
# symbols = ["btcusdt", "ethusdt", "xrpusdt", "trxusdt", "eosusdt", "zecusdt", "ltcusdt",
#            "etcusdt", "etpusdt", "iotusdt", "rrtusdt", "xmrusdt", "dshusdt", "avtusdt",
#            "omgusdt", "sanusdt", "qtmusdt", "edousdt", "btgusdt", "neousdt", "zrxusdt",
#            "tnbusdt", "funusdt", "mnausdt", "sntusdt", "gntusdt"]
#
# # symbols = ["btcusdt"]
#
# for i in range(len(symbols)):
#     dataf = read_data("BITFINEX", symbols[i], '1h', "2017-01-01", "2018-12-21")
#     print(dataf.head())
#     print(dataf.tail())
#     dataf["date_time"] = pd.to_datetime(dataf["date"])
#     df = dataf.resample(rule="4h", on='date_time',label="left").apply(
#                 {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
#                  'volume': 'sum', "amount": "sum"})
#     df["date"] = df.index
#     df.index = range(len(df))
#     print(df.head())
#     print(df.tail())
    # df.to_csv("/Users/wuyong/alldata/original_data/BITFINEX_"+symbols[i]+"_4h_2017-01-01_2018-12-21.csv")
# df["index"] = range(len(df))
# df["date"] = df.index
# df.index = df["index"]
#
# print(df)
# print(len(df))
# print(3828-3393)
# df.to_csv("/Users/wuyong/alldata/factor_writedb/factor_stra_4h/BITMEX_.bxbt_4h_2017-01-01_2018-10-01.csv")

# a = np.zeros((13,))
# print(a)
# print(a+1)
#
# combine_value = pd.Series(np.zeros(10),name="mid_value")
# combine_value[combine_value>-1]=combine_value+1
# print(combine_value)
# timestamp = 1535731198995
#
#
# def timestamp_to_strtime(timestamp):
#     """将 13 位整数的毫秒时间戳转化成本地普通时间 (字符串格式)
#     :param timestamp: 13 位整数的毫秒时间戳 (1456402864242)
#     :return: 返回字符串格式 {str}'2016-02-25 20:21:04.242000'
#     """
#     local_str_time = datetime.fromtimestamp(timestamp / 1000.0).strftime('%Y-%m-%d %H:%M:%S.%f')
#     return local_str_time
#
#
def timestamp_to_datetime(timestamp):
    """将 13 位整数的毫秒时间戳转化成本地普通时间 (datetime 格式)
    :param timestamp: 13 位整数的毫秒时间戳 (1456402864242)
    :return: 返回 datetime 格式 {datetime}2016-02-25 20:21:04.242000
    """
    local_dt_time = datetime.fromtimestamp(timestamp/1000.0)
    return local_dt_time
#
#
# print(timestamp_to_strtime(timestamp))
# print(timestamp_to_datetime(timestamp))

#
# data_results = pd.read_csv("/Users/wuyong/alldata/original_data/result_save_btc.csv", index_col=0)
# data_seconds = pd.read_csv("/Users/wuyong/alldata/original_data/trades_bian_btcusdt_s.csv")
# data_results["price"] = data_seconds["price"]
# data_results["dealtime"] = data_seconds["dealtime"].apply(lambda x: timestamp_to_datetime(x))
# # data_seconds.index = data_seconds["dealtime"].values
# # print(data_seconds.loc[1533052860:1533052920])
# # print(len(data_seconds.loc[1533052860:1533052919]))
# # print(data_seconds["btcnum"])
# print(len(data_results[data_results["btcnum"]>0]))
# # print(data_results[data_results["btcnum"]>0])
# data_test = data_results.iloc[1846970:1847520]
# print(data_test)


# data_results = pd.read_csv("/Users/wuyong/alldata/original_data/trades_bian_btcusdt_s_1.csv",index_col=0)
# # print(data_results.columns)
# # print(len(data_results))
# print(data_results.head(20))

# data_results["dealtime"] = data_results["dealtime"].apply(lambda x: int(x/1000))
# data_results.drop_duplicates(keep="last",inplace=True,subset="dealtime")
# print(data_results.head(20))
# print(data_results.tail())
#
# data_last = pd.DataFrame({"dealtime":range(1543593602,1544134875)})
#
# print(data_last.head())
# print(data_last.tail())
#
# data_last = data_last.merge(data_results,how="left",on="dealtime")
# data_last.fillna(method="ffill",inplace=True)
# print(data_last.head(50))
# data_last.to_csv("/Users/wuyong/alldata/original_data/trades_bian_btcusdt_s_1.csv")


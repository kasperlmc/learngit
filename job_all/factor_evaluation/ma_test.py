# coding=utf-8

import sys
sys.path.append('..')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# from lib.factors_gtja import *
# from lib.myfun import *
import os
import talib as ta

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
# pd.set_option('display.max_rows', None)


symbols_close = [x+"_close" for x in ["btcusdt","ethusdt", "xrpusdt", "eosusdt", "trxusdt"]]
symbols = ['btcusdt',"ethusdt","xrpusdt","trxusdt","eosusdt","zecusdt","ltcusdt",
           "etcusdt","etpusdt","iotusdt","rrtusdt","xmrusdt","dshusdt","avtusdt",
           "omgusdt","sanusdt","qtmusdt","edousdt","btgusdt","neousdt","zrxusdt",
           "tnbusdt","funusdt","mnausdt","sntusdt","gntusdt"]
# symbols=["ethusdt","btcusdt"]

alpha="Alpha.alpha020"

for symbol in symbols:
    data = pd.read_csv('/Users/wuyong/alldata/factor_writedb/factor_stra/' + symbol + "_" + alpha + "_gtja1h" + '.csv',
                       index_col=0)

    if symbol == "btcusdt":
        df = pd.DataFrame({symbol: data[alpha].values, symbol + "_close": data["close"].values,
                           symbol + "_open": data["open"].values}, index=data["date"].values)
    else:
        df_1 = pd.DataFrame({symbol: data[alpha].values, symbol + "_close": data["close"].values,
                             symbol + "_open": data["open"].values}, index=data["date"].values)
        df = pd.concat([df, df_1], axis=1)

df["index"] = range(len(df))
df["date_time"] = df.index
df.index = df["index"]
# print(df[symbols_close].fillna(method="ffill"))
df[symbols_close]=df[symbols_close].fillna(method="ffill")[symbols_close]
# print(df.tail(100))
# print(len(df), len(df[["xrpusdt_close", "xrpusdt_open"]].dropna()))
# df.fillna(inplace=True, method="ffill")
ma_list = [3, 5, 10, 15, 30, 60, 120, 240]
for close in ["ethusdt_close","btcusdt_close","xrpusdt_close","eosusdt_close","trxusdt_close"]:
    # df[close+"com"] = np.zeros(len(df))
    combine_value = pd.Series(np.zeros(len(df)), name="mid_value")
    for i in range(len(ma_list)):
        df[close + str(ma_list[i])] = ta.MA(df[close].values, timeperiod=ma_list[i], matype=0)
        if i == 0:
            combine_value[df[close] > df[close + str(ma_list[i])]] = combine_value + 1
        else:
            combine_value[df[close + str(ma_list[i - 1])] > df[close + str(ma_list[i])]] = combine_value + 1
    df[close + "com"] = combine_value
# print(df.head(20))
print(len(df))
print(df[["trxusdt_closecom"]])
# df=df.iloc[8000:120000]
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax1.plot(df.index,df["ethusdt_close"])
# ax2=ax1.twinx()
# ax2.plot(df.index,df["ethusdt_closecom"],c="r")
# plt.show()


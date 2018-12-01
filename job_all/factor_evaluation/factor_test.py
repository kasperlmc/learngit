# coding=utf-8

import sys
sys.path.append('..')
import pandas as pd
# import matplotlib.pyplot as plt
import numpy as np
from lib.factors_gtja import *
from lib.myfun import *


# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)

symbols = ['btcusdt',"ethusdt","xrpusdt","trxusdt","eosusdt","zecusdt","ltcusdt",
           "etcusdt","etpusdt","iotusdt","rrtusdt","xmrusdt","dshusdt","avtusdt",
           "omgusdt","sanusdt","qtmusdt","edousdt","btgusdt","neousdt","zrxusdt",
           "tnbusdt","funusdt","mnausdt","sntusdt","gntusdt"]

symbols_close = [x+"_close" for x in ["btcusdt","ethusdt", "xrpusdt", "eosusdt", "trxusdt"]]
symbols_param = ["btcusdt","ethusdt", "xrpusdt", "eosusdt", "trxusdt"]

alpha_test = ["Alpha.alpha018"]
print("max")
stat_ls = []
for alpha in alpha_test:
    # 计算出每个alpha的策略指标
    try:
        for symbol in symbols:
            data=pd.read_csv('/Users/wuyong/alldata/factor_writedb/factor_stra_4h/' + symbol + "_" + alpha + "_gtja4h" + '.csv',index_col=0)

            if symbol=="btcusdt":
                df=pd.DataFrame({symbol:data[alpha].values,symbol+"_close":data["close"].values,symbol+"_open":data["open"].values}, index=data["date"].values)
            else:
                df_1=pd.DataFrame({symbol:data[alpha].values,symbol+"_close":data["close"].values,symbol+"_open":data["open"].values}, index=data["date"].values)
                df=pd.concat([df,df_1],axis=1)

        # print(df.head(40))
        # print(df.columns)
        # print(df[["btcusdt"]])
        # df.dropna(inplace=True)
        # print(df.ix["2018-03-01 00:00:00":])
        # df=df.ix["2018-03-01 00:00:00":]
        df["index"] = range(len(df))
        df["date_time"] = df.index
        df.index = df["index"]
        print(alpha)
        df[symbols_close] = df[symbols_close].fillna(method="ffill")[symbols_close]
        # 计算出资产组合对比指标
        dataf = read_data("BITMEX", ".bxbt", '4h', "2017-01-01", "2018-10-01")
        index_values = list(dataf["close"].values)
        index_values = [index_values[0]]*435+index_values
        # print(index_values)
        df["index"] = np.array(index_values)
        # print(df.head())

    except:
        pass




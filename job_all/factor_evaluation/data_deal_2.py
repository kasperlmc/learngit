import sys
sys.path.append('..')

import pandas as pd
import numpy as np

# 显示所有列
pd.set_option('display.max_columns', None)

symbols = ['btcusdt',"ethusdt","xrpusdt","zecusdt","eosusdt","neousdt",
           "ltcusdt","etcusdt","etpusdt","iotusdt","rrtusdt","xmrusdt",
           "dshusdt","avtusdt","omgusdt","sanusdt","qtmusdt","edousdt",
           "btgusdt","trxusdt","zrxusdt","tnbusdt","funusdt", "mnausdt",
           "sntusdt","gntusdt"]
symbols = ["btcusdt"]

alpha_test = ["Alpha.alpha020"]

for alpha in alpha_test:
    for symbol in symbols:
        factor_name = alpha + "_" + "gtja1h"
        fname = '/Users/wuyong/alldata/factor_writedb/factor_stra/' + symbol + '_' + factor_name + '.csv'
        data_mid1 = pd.read_csv(fname,index_col=0)
        data_mid1["date_time"] = pd.to_datetime(data_mid1["date"])
        df = data_mid1.resample(rule="4h", on='date_time',label="left").apply(
            {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
             'volume': 'sum',"amount":"sum"})
        df["index"] = range(len(df))
        df["date"] = df.index
        df.index = df["index"]
        print(len(df))
        print(df.tail(10))
        # print(df.head(10))
        # df.to_csv("/Users/wuyong/alldata/factor_writedb/factor_stra_4h/" + "BITFINEX_" + symbol
        #           + "_4h_2017-01-01_2018-10-01" + ".csv")
        break



















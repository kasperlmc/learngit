# coding=utf-8

import sys
sys.path.append('..')
import pandas as pd
import numpy as np
import talib as ta
import time
from numba import jit
np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)
# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 滑点设置
slippage = 0.000


def ts_sum(df, window=10):
    return df.rolling(window).sum()


def ts_max(df, window=10):
    return df.rolling(window).max()


def ts_min(df ,window=10):
    return df.rolling(window).min()


def ts_lowday(df, window=10):
    return (window-1)-df.rolling(window).apply(np.argmin)


coin_list = ["btcusdt", "xrpusdt", "eosusdt", "ethusdt"]
columns = ["close", "position"]

for i in range(len(coin_list)):
    # 数据准备，数据说明
    data_zb = pd.read_csv("/Users/wuyong/alldata/original_data/" + coin_list[i] + "_trade" + ".csv", index_col=0)
    coin_name = coin_list[i]
    print(i)
    print(data_zb.head(10))

    if i == 0:
        data_all = data_zb
        data_all.columns = ["tickid"]+[x + "_" + coin_name for x in columns]
    else:
        data_temp = data_zb
        data_temp.columns = ["tickid"]+[x + "_" + coin_name for x in columns]
        data_all = data_all.merge(data_temp, how="left", on="tickid")

print(data_all.head(10))
print(data_all.tail(100))

data_all.dropna(inplace=True)

data_all.to_csv("/Users/wuyong/alldata/original_data/celue_all_coindata.csv")



































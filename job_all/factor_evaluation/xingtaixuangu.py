# coding=utf-8

import sys
sys.path.append('..')
import pandas as pd
import numpy as np
import talib as ta
import time
from numba import jit
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw


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


# 数据准备，导入币安各个币对
data_zb = pd.read_csv("/Users/wuyong/alldata/original_data/trades_BIAN_btcusdt_s_3mon.csv", index_col=0)
data_zb["tickid"] = data_zb["dealtime"]
data_zb["close_s"] = data_zb["price"]
data_k = pd.read_csv("/Users/wuyong/alldata/original_data/trades_bian_btcusdt_m_3mon.csv", index_col=0)
# data_k = data_k.tail(1000)
standard_w = [7,6,5,4,3,2,1,2,3,4,5,6,7,6,5,4,3,2,1,2,3,4,5,6,7]
standard_w = np.array(standard_w)
standard_w = (standard_w-standard_w.mean())/standard_w.std()
# print((standard_w-standard_w.mean())/standard_w.std())
data_k["ema7"] = ta.MA(data_k["close"].values, timeperiod=7)
print(data_k.tail(20))
# print(data_k.loc[132621:132675],len(data_k.loc[132621:132675]))
test = data_k.loc[132621:132675]["ema7"].values
test = (test-test.mean())/test.std()
distance, path = fastdtw(standard_w, test, dist=euclidean)
print(distance)


def cal_distance(y, standard_w=standard_w):
    # print(y)
    mid_value = (y.values-y.values.mean())/y.values.std()
    distance, path = fastdtw(standard_w, mid_value, dist=euclidean)
    return distance


# data_k["dis"] = data_k["ema7"].rolling(window=55).apply(cal_distance,raw=False)
# print(data_k[["date","dis"]])
data_k_test = data_k
# print(data_k_test)

data_k_test["dis"] = data_k_test["close"].rolling(window=55).apply(cal_distance,raw=False)
# print(data_k_test[["date", "dis"]])
data_k_test.to_csv("/Users/wuyong/alldata/original_data/trades_bian_btcusdt_m_3mon_dis.csv")









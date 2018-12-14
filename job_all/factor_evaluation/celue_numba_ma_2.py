# coding=utf-8

import sys
sys.path.append('..')
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
# from lib.myfun import *
import os
import talib as ta
import time
from numba import jit
import logging
import copy
import datetime
import mpl_finance as mpf
from matplotlib.pylab import date2num
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


# 数据准备，数据说明
data_zb = pd.read_csv("/Users/wuyong/alldata/original_data/trades_bian_xrpusdt_s_2.csv", index_col=0)
# print(data_zb.head())
data_zb["tickid"] = data_zb["dealtime"]
# data_zb["tickid"] = data_zb.index
data_zb["close"] = data_zb["price"]
# print(data_zb.head(30))
# print(len(data_zb))
data_k = pd.read_csv("/Users/wuyong/alldata/original_data/trades_bian_xrpusdt_m_2.csv", index_col=0)
# print(data_k.head(20))
# atr = ta.ATR(data_k['high'].values, data_k['low'].values, data_k['close'].values, timeperiod=90)
# data_k["atr_90"] = atr
atr = ta.ATR(data_k['high'].values, data_k['low'].values, data_k['close'].values, timeperiod=7)
data_k["atr_7"] = atr
data_k["growth"] = data_k["close"]-data_k["open"]
data_k["growth"] = data_k["growth"].apply(lambda x: 1 if x > 0 else 0)
data_k["growth"] = ts_sum(data_k["growth"])
data_k["high_35"] = ts_max(data_k["high"], window=55)
data_k["low_30"] = ts_min(data_k["low"],window=30)
data_k["ma23"] = ta.MA(data_k["close"].values, timeperiod=23, matype=0)
data_k["ma90"] = ta.MA(data_k["close"].values, timeperiod=90, matype=0)
# data_k["diff"] = data_k["ma7"]-data_k["ma25"]
# data_k["diff"] = data_k["diff"].apply(lambda x: 1 if x > 0 else x)
# data_k["diff"] = data_k["diff"].apply(lambda x: -1 if x < 0 else x)
# data_k["diff"] = data_k["diff"].diff()
# data_k["diff"] = data_k["diff"].apply(lambda x: 0 if x > 0 else x)
# data_k["diff"] = data_k["diff"].apply(lambda x: 1 if x < 0 else x)
data_zb = data_zb[["tickid", "close"]]
data_zb["close_speed_6"] = (data_zb["close"]-data_zb["close"].shift(6))/data_zb["close"].shift(6)
data_zb["close_speed_12"] = (data_zb["close"].shift(6)-data_zb["close"].shift(12))/data_zb["close"].shift(12)
print(data_zb.head(20))
print(data_zb.tail(10))
data_zb = data_zb[data_zb["tickid"] > 1543680000]
# print(len(data_zb))
data_k = data_k[["tickid", "open", "high", "close", "low", "growth", "high_35", "ma23", "ma90", "low_30"]]
# data_k["ratio"] = 2*data_k["atr"]/data_k["close"]
# print(data_k.head(100))
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax1.plot(data_k.index,data_k["close"])
# ax2=ax1.twinx()
# ax2.plot(data_k.index,data_k["ratio"],c="r")
# plt.show()

print(data_k.head(50))
# print(data_k.values)
# value_test = data_k.values
# print(type(value_test))
# print(value_test.shape)
# print(value_test[0])
# print(value_test[0][0])
# print(data_zb.values.shape)
# print(value_test[(value_test[:,0]<1541693640) & (value_test[:,0]>1541693520)][:, 5].sum())
# print(value_test[value_test[:,0]<1543694620][-4:][:,3].max())
# print(data_zb.values[5-4:5])


@jit()
def numba_celue(data_zb, data_k):

    cash_list = np.zeros(1)
    asset_list = np.zeros(1)
    btcnum_list = np.zeros(1)
    cash_list[0] = 10000.0
    asset_list[0] = 10000.0
    btcnum_list[0] = 0.0
    buy_time = 0
    update_low = 10000
    update_max = 10000
    tradetimes = 0
    profittimes = 0
    profitnum = 0
    losstimes = 0
    lossnum = 0
    buy_price = 0
    max_temp = 0
    low_temp = 0
    for n in range(1, len(data_zb)):
        if len(data_k[data_k[:, 0] < data_zb[n][0]]) < 35:
            cash_list = np.append(cash_list,cash_list[-1])
            btcnum_list = np.append(btcnum_list,btcnum_list[-1])
            asset_list = np.append(asset_list,asset_list[-1])
        else:
            if btcnum_list[n-1] == 0.0:
                if (data_k[data_k[:, 0] < data_zb[n][0]][-7:][:, 2].max() == data_k[data_k[:, 0] < data_zb[n][0]][-7:][:, 6].max()) and \
                        (data_k[data_k[:, 0] < data_zb[n][0]][-1][8] >= data_k[data_k[:, 0] < data_zb[n][0]][-2][8]*1.0001) and \
                        (data_zb[n][1] <= data_k[data_k[:, 0] < data_zb[n][0]][-1][7]):
                    pass
                    print("开仓买入")
                    print("买入时点为%s，买入时点价格为%s" % (data_zb[n][0], data_zb[n][1]))
                    buy_price = data_zb[n][1] * (1 + slippage)  # 按照当前逐笔数据的价格乘以一个滑点买入币对
                    btcnum = cash_list[-1] / buy_price  # 买入的数量根据现有的资金量计算出来
                    print("买入价格为:%s" % buy_price)
                    btcnum_list = np.append(btcnum_list, btcnum)  # 币对持仓数目列表记录该次交易买入的数量
                    cash_list = np.append(cash_list, 0)  # 币对现金账户列表记录当前账户所有的现金数量
                    asset_list = np.append(asset_list, cash_list[-1]+btcnum_list[-1]*data_zb[n][1])  # 资产列表记录所有的总资产=现金数目+币对数目*当前逐笔数据价格
                    buy_time = n
                    max_temp = data_k[data_k[:, 0] < data_zb[n][0]][-7:][:, 2].max()
                    low_temp = data_k[data_k[:, 0] < data_zb[n][0]][-1][9]
                else:
                    cash_list = np.append(cash_list, cash_list[-1])
                    btcnum_list = np.append(btcnum_list, btcnum_list[-1])
                    asset_list = np.append(asset_list, asset_list[-1])

            else:
                if data_k[data_k[:, 0] < data_zb[n][0]][-7:][:, 2].max() > data_zb[buy_time:n][:, 1].max():
                    update_low = data_k[data_k[:, 0] < data_zb[n][0]][-7:][:, 4].min()
                    update_max = data_k[data_k[:, 0] < data_zb[n][0]][-7:][:, 2].max()
                if data_zb[n][1] < (max(max_temp, data_zb[buy_time:n][:, 1].max())+low_temp)*0.5:
                    print("平仓卖出")
                    print(data_zb[buy_time:n][:, 1].max(),data_k[data_k[:, 0] < data_zb[buy_time][0]][-3:][:, 4].min())
                    print((data_zb[buy_time:n][:, 1].max()+data_k[data_k[:, 0] < data_zb[buy_time][0]][-3:][:, 4].min())*0.5)
                    print("卖出时点为%s，卖出时点价格为%s" % (data_zb[n][0], data_zb[n][1]))
                    sell_price = data_zb[n][1] * (1 - slippage)  # 按照当前逐笔数据的价格加上滑点和手续费卖出所有持仓
                    print("平仓价格:%s" % sell_price)
                    print("当前时点价格为%s,止损价格为%s" % (data_zb[n][1],(data_zb[buy_time][1]+data_k[data_k[:, 0] < data_zb[n][0]][-3:][:, 4].min())/2))
                    print("最近K线的短期atr为%s，六天前的两倍atr为%s" % (data_k[data_k[:, 0] < data_zb[n][0]][-1][7], data_k[data_k[:, 0] < data_zb[n][0]][-6][7]*2))
                    cash_get = sell_price * btcnum_list[-1]  # 卖出所有持仓获得的现金
                    cash_list = np.append(cash_list, cash_get)  # 现金账户列表更新
                    btcnum_list = np.append(btcnum_list, 0)  # 币对数目账户列表更新
                    asset_list = np.append(asset_list, cash_list[-1]+btcnum_list[-1]*data_zb[n][1])  # 资产账户列表更新
                    update_low = 10000
                    update_max = 10000
                    tradetimes += 1
                    if sell_price > buy_price:
                        profittimes += 1
                        profitnum += btcnum_list[-2]*(sell_price-buy_price)
                    elif sell_price < buy_price:
                        losstimes += 1
                        lossnum += btcnum_list[-2]*(sell_price-buy_price)
                    print(tradetimes, profitnum, profittimes, lossnum, losstimes)
                    print(asset_list[-1])
                else:
                    cash_list = np.append(cash_list, cash_list[-1])
                    btcnum_list = np.append(btcnum_list, btcnum_list[-1])
                    asset_list = np.append(asset_list, cash_list[-1]+btcnum_list[-1]*data_zb[n][1])  # 资产账户列表更新
    return cash_list, asset_list, btcnum_list, tradetimes, profitnum, profittimes, lossnum, losstimes


print("xrpusdt—————————————10——————04—————————————02—————01—————————")
start = time.time()
# print(numba_celue(data_zb.values,data_k.values))
cash_list,asset_list,btcnum_list,tradetimes,profitnum,profittimes,lossnum,losstimes = numba_celue(data_zb.values,data_k.values)
end = time.time()
print("Elapsed (with compilation) = %s" % (end - start))
df_result = pd.DataFrame({"cash": cash_list, "asset": asset_list, "btcnum": btcnum_list})
print(df_result.head(20))
print(df_result.tail(20))

























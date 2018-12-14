# coding=utf-8

import sys


class TailRecurseException(Exception):
    def __init__(self, args, kwargs):
        self.args = args
        self.kwargs = kwargs


def tail_call_optimized(g):
    """
    This function decorates a function with tail call
    optimization. It does this by throwing an exception
    if it is it's own grandparent, and catching such
    exceptions to fake the tail call optimization.

    This function fails if the decorated
    function recurses in a non-tail context.
    """

    def func(*args, **kwargs):
        f = sys._getframe()
        # 为什么是grandparent, 函数默认的第一层递归是父调用,
        # 对于尾递归, 不希望产生新的函数调用(即:祖父调用),
        # 所以这里抛出异常, 拿到参数, 退出被修饰函数的递归调用栈!(后面有动图分析)
        if f.f_back and f.f_back.f_back and f.f_back.f_back.f_code == f.f_code:
            # 抛出异常
            raise TailRecurseException(args, kwargs)
        else:
            while 1:
                try:
                    return g(*args, **kwargs)
                except TailRecurseException as e:
                    args = e.args
                    kwargs = e.kwargs
    func.__doc__ = g.__doc__
    return func


sys.path.append('..')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from lib.myfun import *
import os
import talib as ta
import logging
import copy


# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)

slippage = 0.002

dataf_1m = read_data("BIAN", "btcusdt", '1m', "2018-06-01", "2018-10-01")
# print(dataf_1m.head(30))
# dataf_5m = read_data("BIAN", "btcusdt", '5m', "2018-06-01", "2018-10-01")
# print(dataf_5m.head(10))

# print(dataf_5m[dataf_5m["tickid"]<1527784080].tail(5))
print(len(dataf_1m))

data_zb = dataf_1m[["close", "tickid", "date"]]
data_zb["date_time"] = pd.to_datetime(data_zb["date"])
# print(data_zb.head(20))

# df = data_zb.resample(rule="5min", on='date_time', label="right", closed="right").apply({"close": "first"})
# print(df.head(30))

data_k = data_zb.resample(rule="60min", on='date_time', label="right", closed="right").apply({"close": "last"})

data_k["open"] = data_zb.resample(rule="60min", on='date_time', label="right", closed="right").apply({"close": "first"})["close"]

data_k["high"] = data_zb.resample(rule="60min", on='date_time', label="right", closed="right").apply({"close": "max"})["close"]

data_k["low"] = data_zb.resample(rule="60min", on='date_time', label="right", closed="right").apply({"close": "min"})["close"]

data_k["tickid"] = data_zb.resample(rule="60min", on='date_time', label="right", closed="right").apply({"tickid": "last"})["tickid"]
data_k["index"] = range(len(data_k))
data_k["date_time"] = data_k.index
data_k.index = data_k["index"]
del data_k["index"]
# print(data_k.head(30))
# print(data_k[data_k["tickid"]<1527783360])
print(data_zb.iloc[-1])
print(data_zb.head(20))
print(data_k.head(20))
print(data_k.head(10)["high"].max())


@tail_call_optimized
def strategy_func(data_zb, data_k, n=1, cash_list=[10000], asset_list=[10000], btcnum_list=[0], date_list=["2018-06-01 00:01:00"]):
    if n == len(data_zb):
        return cash_list, asset_list, btcnum_list, date_list

    data_tem_zb = data_zb.iloc[n]
    data_tem_k = data_k[data_k["tickid"] < data_tem_zb["tickid"]]
    date_list.append(data_tem_zb["date"])
    if len(data_tem_k) < 30 or n % 6 != 0:
        cash_list.append(cash_list[-1])
        btcnum_list.append(btcnum_list[-1])
        asset_list.append(cash_list[-1]+btcnum_list[-1]*data_tem_zb["close"])
        return strategy_func(data_zb, data_k, n+1, cash_list, asset_list, btcnum_list, date_list)

    else:
        data_tem_k_tail = data_tem_k.tail(30)
        high_20k = data_tem_k_tail.iloc[-20:]["high"].max()

        if btcnum_list[-1] == 0:
            if (data_tem_zb["close"] > high_20k) and ((data_tem_zb["close"]-data_zb.iloc[n-6]["close"])/data_zb.iloc[n-6]["close"] > 0.01) and \
                    (data_tem_k_tail.iloc[-1]["close"] > data_tem_k_tail.iloc[-7]["close"]):
                # print("开仓买入")
                # print(n)
                # print(data_tem_k_tail)
                # print(data_tem_zb)
                # print(high_20k, data_tem_zb["close"])
                # print(data_zb.iloc[n-6])
                buy_price = data_tem_zb["close"]*(1+slippage)
                btcnum = cash_list[-1]/buy_price
                btcnum_list.append(btcnum)
                cash_list.append(0)
                asset_list.append(cash_list[-1]+btcnum_list[-1]*data_tem_zb["close"])
                return strategy_func(data_zb, data_k, n+1, cash_list, asset_list, btcnum_list, date_list)
            else:
                cash_list.append(cash_list[-1])
                btcnum_list.append(btcnum_list[-1])
                asset_list.append(cash_list[-1] + btcnum_list[-1] * data_tem_zb["close"])
                return strategy_func(data_zb, data_k, n + 1, cash_list, asset_list, btcnum_list, date_list)

        else:
            atr = ta.ATR(data_tem_k_tail['high'].values, data_tem_k_tail['low'].values, data_tem_k_tail['close'].values, timeperiod=14)[-1]
            data_temp_last_k = data_tem_k_tail.iloc[-1]
            # print(data_tem_k_tail.iloc[-4:]["high"].max(), data_tem_k_tail.iloc[-5]["high"])
            if (data_tem_zb["close"] < data_temp_last_k["close"] - 2*atr) or (data_tem_k_tail.iloc[-4:]["high"].max() < data_tem_k_tail.iloc[-5]["high"]):
                # print("平仓卖出")
                # print(n)
                # print(data_tem_zb)
                # print(atr)
                # print(data_tem_zb["close"], data_temp_last_k["close"])
                # print(data_tem_k_tail.iloc[-4:])
                # print(data_tem_k_tail.iloc[-5])
                sell_price = data_tem_zb["close"]*(1-slippage)
                cash_get = sell_price*btcnum_list[-1]
                cash_list.append(cash_get)
                btcnum_list.append(0)
                asset_list.append(cash_list[-1] + btcnum_list[-1] * data_tem_zb["close"])
                return strategy_func(data_zb, data_k, n + 1, cash_list, asset_list, btcnum_list, date_list)

            else:
                cash_list.append(cash_list[-1])
                btcnum_list.append(btcnum_list[-1])
                asset_list.append(cash_list[-1] + btcnum_list[-1] * data_tem_zb["close"])
                return strategy_func(data_zb, data_k, n + 1, cash_list, asset_list, btcnum_list, date_list)


# print(strategy_func(data_zb, data_k))

cash_list, asset_list, btcnum_list, date_list = strategy_func(data_zb, data_k)

df = pd.DataFrame({"cash": cash_list, "asset": asset_list, "btcnum": btcnum_list, "date": date_list})

print(df.iloc[-1])



























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
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from lib.myfun import *
import os
import talib as ta
import logging
import copy
import datetime
import mpl_finance as mpf
from matplotlib.pylab import date2num

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 滑点设置
slippage = 0.002
# 数据准备，数据说明
data_zb = pd.read_csv("/Users/wuyong/alldata/original_data/trades_bian_btcusdt_s.csv")
data_zb["tickid"] = data_zb["dealtime"]
data_zb["close"] = data_zb["price"]
# print(data_zb.head(130))
# print(len(data_zb))
data_k = pd.read_csv("/Users/wuyong/alldata/original_data/trades_bian_btcusdt_m.csv",index_col=0)
print(data_k.head(20))
print(len(data_k))
print(type(data_k["date"].values[0]))

data_k_all = data_k.iloc[30749:30784]
data_k_price = data_k_all[["open","high","low","close"]]


# def date_to_num(dates):
#     num_time = []
#     for date in dates:
#         date_time = datetime.datetime.strptime(date,'%Y-%m-%d %H:%M:%S')
#         num_date = date2num(date_time)*10000
#         num_time.append(num_date)
#     return num_time
#
#
# num_list = date_to_num(data_k.head(50)["date"].values)
# print(num_list)
candleData = np.column_stack([list(range(len(data_k_price))), data_k_price])
print(candleData)
fig = plt.figure(figsize=(10, 6))
ax = fig.add_axes([0.1, 0.3, 0.8, 0.6])
mpf.candlestick_ohlc(ax, candleData, width=0.5, colorup='r', colordown='b')
plt.grid(True)
plt.xticks(list(range(len(data_k_price))),list(data_k_all["date"].values))
plt.xticks(rotation=85)
plt.show()
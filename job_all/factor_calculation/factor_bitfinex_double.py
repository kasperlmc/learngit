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
from lib.factors_gtja import *
from lib.myfun import *
import os
import talib as ta
import logging

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)

slippage = 0.002

symbols = ["btcusdt", "ethusdt", "xrpusdt", "trxusdt", "eosusdt", "zecusdt", "ltcusdt",
           "etcusdt", "etpusdt", "iotusdt", "rrtusdt", "xmrusdt", "dshusdt", "avtusdt",
           "omgusdt", "sanusdt", "qtmusdt", "edousdt", "btgusdt", "neousdt", "zrxusdt",
           "tnbusdt", "funusdt", "mnausdt", "sntusdt", "gntusdt"]


def double_factor_celue(data_celue, datetime_celue):
    cash_list = np.zeros(len(data_celue))
    asset_list = np.zeros(len(data_celue))
    btcnum_list = np.zeros(len(data_celue))
    ethnum_list = np.zeros(len(data_celue))
    date_list = []
    position_list = np.zeros(len(data_celue))
    date_list.append(datetime_celue[0])
    cash_list[0] = 10000.0
    asset_list[0] = 10000.0
    btcnum_list[0] = 0.0
    ethnum_list[0] = 0.0
    tradetimes = 0
    profittimes = 0
    profitnum = 0
    losstimes = 0
    lossnum = 0
    buy_price = 0
    tupo_l = 0
    for n in range(1, len(data_celue)):
        date_list.append(datetime_celue[n])
        if btcnum_list[n-1] != 0:
            if np.abs(data_celue[n-1][0]-data_celue[n-1][3]) < 25:
                trade_price_btc = data_celue[n][2]*(1-np.sign(btcnum_list[n-1])*slippage)
                trade_price_eth = data_celue[n][5]*(1-np.sign(ethnum_list[n-1])*slippage)
                cash_list[n] = cash_list[n-1] + btcnum_list[n-1]*trade_price_btc + ethnum_list[n-1]*trade_price_eth
                btcnum_list[n] = 0.0
                ethnum_list[n] = 0.0
                asset_list[n] = cash_list[n]
            else:
                cash_list[n] = cash_list[n-1]
                btcnum_list[n] = btcnum_list[n-1]
                ethnum_list[n] = ethnum_list[n-1]
                asset_list[n] = cash_list[n] + btcnum_list[n]*data_celue[n][1] + ethnum_list[n]*data_celue[n][4]
        else:
            if np.abs(data_celue[n-1][0] - data_celue[n-1][3]) >= 25:
                trade_price_btc = data_celue[n][2]*(1 + np.sign(data_celue[n-1][0]-data_celue[n-1][3])*slippage)
                trade_price_eth = data_celue[n][5]*(1 + np.sign(data_celue[n-1][3]-data_celue[n-1][0])*slippage)
                btcnum_list[n] = np.sign(data_celue[n-1][0]-data_celue[n-1][3])*(cash_list[n-1]/trade_price_btc)
                ethnum_list[n] = np.sign(data_celue[n-1][3]-data_celue[n-1][0])*(cash_list[n-1]/trade_price_eth)
                cash_list[n] = cash_list[n-1]
                asset_list[n] = cash_list[n] + btcnum_list[n]*data_celue[n][1] + ethnum_list[n]*data_celue[n][4]
            else:
                btcnum_list[n] = btcnum_list[n-1]
                ethnum_list[n] = ethnum_list[n-1]
                cash_list[n] = cash_list[n-1]
                asset_list[n] = asset_list[n-1]
    return cash_list, asset_list, btcnum_list, ethnum_list, date_list


a = list(range(1, 202))
alpha_test = []
for x in a:
    if x < 10:
        alpha_test.append("Alpha.alpha00"+str(x))
    elif 10 < x < 100:
        alpha_test.append("Alpha.alpha0"+str(x))
    else:
        alpha_test.append("Alpha.alpha" + str(x))

# alpha_test = ["Alpha.alpha040","Alpha.alpha029","Alpha.alpha043","Alpha.alpha057"]
# alpha_test = ["Alpha.alpha040"]

print("max")
stat_ls = []
for alpha in alpha_test:
    # 计算出每个alpha的策略指标
    try:
        for symbol in symbols:
            data = pd.read_csv('/Users/wuyong/alldata/factor_writedb/factor_stra_4h/BITFINEX_' + symbol + "_" + alpha + "_gtja4h" + '.csv', index_col=0)
            # print(data.head())
            # exit()
            if symbol == "btcusdt":
                df = pd.DataFrame({symbol: data[alpha].values, symbol+"_close": data["close"].values, symbol+"_open":data["open"].values}, index=data["date"].values)
            elif symbol == "ethusdt":
                df_1 = pd.DataFrame({symbol: data[alpha].values, symbol + "_close": data["close"].values,
                                     symbol + "_open": data["open"].values}, index=data["date"].values)
                df = df.merge(df_1,left_index=True,right_index=True,how="left")
            else:
                df_1 = pd.DataFrame({symbol: data[alpha].values}, index=data["date"].values)
                df = df.merge(df_1, left_index=True, right_index=True, how="left")
        df[symbols] = df[symbols].rank(axis=1, numeric_only=True, na_option="keep")
        # print(df.head())
        # print(df[symbols].rank(axis=1, numeric_only=True, na_option="keep").tail())
        data_celue_values = df.iloc[:, :6].values
        # print(data_celue_values)
        date_list_values = df.index.values
        # print(date_list_values)
        cash_list, asset_list, btcnum_list, ethnum_list, date_list = double_factor_celue(data_celue_values, date_list_values)
        # print(cash_list)
        # print(asset_list)
        # print(btcnum_list)
        # print(ethnum_list)
        df_result = pd.DataFrame({"cash":cash_list, "asset":asset_list, "btcnum":btcnum_list, "ethnum":ethnum_list}, index=date_list)
        print(alpha)
        print(df_result.tail())
    except FileNotFoundError:
        pass



















































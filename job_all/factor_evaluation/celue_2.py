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
# from lib.myfun import *
import os
import talib as ta
from numba import jit
import logging
import copy
import time
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
data_zb = pd.read_csv("/Users/wuyong/alldata/original_data/trades_bian_btcusdt_s_1.csv",index_col=0)
data_zb["tickid"] = data_zb["dealtime"]
data_zb["close"] = data_zb["price"]
print(data_zb.head(30))
# print(len(data_zb))
data_k = pd.read_csv("/Users/wuyong/alldata/original_data/trades_bian_btcusdt_m_1.csv",index_col=0)
print(data_k.head(20))
data_k["growth"] = data_k["close"]-data_k["open"]
data_k["growth"] = data_k["growth"].apply(lambda x: 1 if x > 0 else 0)
# print(len(data_k))
# print(type(data_k["date"].values[0]))
#
# data_k_all = data_k.iloc[18108:18141]
# data_k_price = data_k_all[["open","high","low","close"]]
#
#
# # def date_to_num(dates):
# #     num_time = []
# #     for date in dates:
# #         date_time = datetime.datetime.strptime(date,'%Y-%m-%d %H:%M:%S')
# #         num_date = date2num(date_time)*10000
# #         num_time.append(num_date)
# #     return num_time
# #
# #
# # num_list = date_to_num(data_k.head(50)["date"].values)
# # print(num_list)
# candleData = np.column_stack([list(range(len(data_k_price))), data_k_price])
# print(candleData)
# fig = plt.figure(figsize=(10, 6))
# ax = fig.add_axes([0.1, 0.3, 0.8, 0.6])
# mpf.candlestick_ohlc(ax, candleData, width=0.5, colorup='r', colordown='b')
# plt.grid(True)
# plt.xticks(list(range(len(data_k_price))),list(data_k_all["date"].values))
# plt.xticks(rotation=85)
# plt.show()


# mat_wdyx = data_k_test.as_matrix()
# num_time = date_to_num(mat_wdyx[:, 0])
# mat_wdyx[:, 0] = num_time
# print(mat_wdyx)


# fig, ax = plt.subplots(figsize=(15,5))
# fig.subplots_adjust(bottom=0.5)
# mpf.candlestick_ochl(ax, mat_wdyx, width=0.6, colorup='g', colordown='r', alpha=1.0)
# plt.grid(True)
# # 设置日期刻度旋转的角度
# plt.xticks(rotation=30)
# plt.title('wanda yuanxian 17')
# plt.xlabel('Date')
# plt.ylabel('Price')
# # x轴的刻度为日期
# ax.xaxis_date()
# plt.show()


@tail_call_optimized
def strategy_func(data_zb, data_k, n=1, cash_list=[10000], asset_list=[10000], btcnum_list=[0]):
    """

    :param data_zb: 逐笔数据
    :param data_k: K线数据
    :param n: 数据的行号，可以用来指代当前的时间，n从1开始
    :param cash_list: 现金账户列表
    :param asset_list: 资产账户列表
    :param btcnum_list: 数字货币持仓个数列表
    :return:
    """
    if n == len(data_zb):  # 递归的终结点
        return cash_list, asset_list, btcnum_list

    data_tem_zb = data_zb.iloc[n]  # 获取当前时间点的逐笔数据
    # print(n)  # 相当于程序的进度条
    data_tem_k = data_k[data_k["tickid"] < data_tem_zb["tickid"]]  # 获取该时间点之前的所有K线数据
    if len(data_tem_k) < 30:  # 如果能获得的最近所有的K线不足30根，就不执行交易
        cash_list.append(cash_list[-1])
        btcnum_list.append(btcnum_list[-1])
        asset_list.append(cash_list[-1]+btcnum_list[-1]*data_tem_zb["close"])
        return strategy_func(data_zb, data_k, n+1, cash_list, asset_list, btcnum_list)

    else:
        data_tem_k_tail = data_tem_k.tail(30)  # 获取最近的30根K线数据
        high_15k = data_tem_k_tail.iloc[-15:]["high"].max()  # 获取最近20K线数据中最高价的最大值
        # print(n)
        # print(high_20k,data_tem_zb["close"],data_zb.iloc[n-12]["close"])
        # print(data_tem_k_tail.iloc[-1]["close"],data_tem_k_tail.iloc[-7]["close"])

        if btcnum_list[-1] == 0:  # 如果当前是空仓
            # 下面有四个条件， 第一个是逐笔数据的价格大于最近15K线数据中最高价的最大值
            # 第二个条件为，（当前逐笔数据的价格-6秒之前逐笔数据价格）/6秒之前逐笔数据价格>0.02
            # 第三个条件为，（6秒前逐笔数据的价格-12秒之前逐笔数据价格）/12秒之前逐笔数据价格>0.01
            # 第四个条件是，最近十K线的阳线数目小于或等于3根
            if (data_tem_zb["close"] > high_15k) and \
                    ((data_tem_zb["close"]-data_zb.iloc[n-6]["close"])/data_zb.iloc[n-6]["close"] > 0.002) and \
                    ((data_tem_zb.iloc[n-6]["close"]-data_zb.iloc[n-12]["close"])/data_zb.iloc[n-12]["close"] > 0.001) and \
                    (data_tem_k_tail.iloc[-10:]["growth"].sum() < 4):
                print("开仓买入")
                print(n)
                print(data_tem_k_tail)
                print(data_tem_zb)
                print(high_15k, data_tem_zb["close"])
                print(data_zb.iloc[n-6])
                buy_price = data_tem_zb["close"]*(1+slippage)  # 按照当前逐笔数据的价格乘以一个滑点买入币对
                btcnum = cash_list[-1]/buy_price  # 买入的数量根据现有的资金量计算出来
                btcnum_list.append(btcnum)  # 币对持仓数目列表记录该次交易买入的数量
                cash_list.append(0)  # 币对现金账户列表记录当前账户所有的现金数量
                asset_list.append(cash_list[-1]+btcnum_list[-1]*data_tem_zb["close"])  # 资产列表记录所有的总资产=现金数目+币对数目*当前逐笔数据价格
                return strategy_func(data_zb, data_k, n+1, cash_list, asset_list, btcnum_list)
            else:
                # 如果不同时满足上述四个交易条件，不进行任何交易，三个账户列表记录列表中的最后一个值
                cash_list.append(cash_list[-1])
                btcnum_list.append(btcnum_list[-1])
                asset_list.append(cash_list[-1] + btcnum_list[-1] * data_tem_zb["close"])
                return strategy_func(data_zb, data_k, n + 1, cash_list, asset_list, btcnum_list)

        else:
            # 计算atr
            atr = ta.ATR(data_tem_k_tail['high'].values, data_tem_k_tail['low'].values, data_tem_k_tail['close'].values, timeperiod=14)[-1]
            data_temp_last_k = data_tem_k_tail.iloc[-1]  # 获取最近一根K线的数据
            # print(n)
            # print(atr)
            # print(data_tem_k_tail.iloc[-4:]["high"].max(), data_tem_k_tail.iloc[-5]["high"])

            # 如果当前逐笔数据的价格比最近840秒的最高价减去两个atr要小，或者小于最近840秒的最高价的99.7%，则平仓
            if data_tem_zb["close"] < min(data_zb.iloc[n-840:n]["close"].max()-2*atr,data_zb.iloc[n-840:n]["close"].max()*0.997):
                print("平仓卖出")
                print(n)
                print(data_tem_zb)
                print(atr)
                print(data_tem_zb["close"], data_temp_last_k["close"])
                print(data_tem_k_tail.iloc[-4:])
                print(data_tem_k_tail.iloc[-5])
                sell_price = data_tem_zb["close"]*(1-slippage)  # 按照当前逐笔数据的价格加上滑点和手续费卖出所有持仓
                cash_get = sell_price*btcnum_list[-1]  # 卖出所有持仓获得的现金
                cash_list.append(cash_get)  # 现金账户列表更新
                btcnum_list.append(0)  # 币对数目账户列表更新
                asset_list.append(cash_list[-1] + btcnum_list[-1] * data_tem_zb["close"])  # 资产账户列表更新
                return strategy_func(data_zb, data_k, n + 1, cash_list, asset_list, btcnum_list)

            else:
                # 如果上述条件都不满足，保持现有持仓不变
                cash_list.append(cash_list[-1])  # 现金账户列表记录最近一次现金数额
                btcnum_list.append(btcnum_list[-1])  # 币对数目列表记录最近一次币对数目
                asset_list.append(cash_list[-1] + btcnum_list[-1] * data_tem_zb["close"])  # 资产数目列表记录按最新价格计算的资产数目
                return strategy_func(data_zb, data_k, n + 1, cash_list, asset_list, btcnum_list)


# start = time.time()
# cash_list, asset_list, btcnum_list = strategy_func(data_zb, data_k)
# end = time.time()
# print("Elapsed (with compilation) = %s" % (end - start))
#
# df = pd.DataFrame({"cash": cash_list, "asset": asset_list, "btcnum": btcnum_list})
#
# print(df.iloc[-1])

# df.to_csv("/Users/wuyong/alldata/original_data/result_save_btc_1.csv")





























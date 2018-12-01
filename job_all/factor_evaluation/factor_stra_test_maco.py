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
# import logging

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)

slippage = 0.00125


symbols = ['btcusdt',"ethusdt","xrpusdt","trxusdt","eosusdt","zecusdt","ltcusdt",
           "etcusdt","etpusdt","iotusdt","rrtusdt","xmrusdt","dshusdt","avtusdt",
           "omgusdt","sanusdt","qtmusdt","edousdt","btgusdt","neousdt","zrxusdt",
           "tnbusdt","funusdt","mnausdt","sntusdt","gntusdt"]

symbols_close = [x+"_close" for x in ["btcusdt","ethusdt", "xrpusdt", "eosusdt", "trxusdt"]]


@tail_call_optimized
def multiple_factor(df,cash_list=[10000],asset_list=[10000],buy_list=[np.nan],btcnum_list=[0],n=1,close_list=[0],max_value_list=[0],posittion=1, win_times=0, price_list=[]):
    """
    此函数为该策略具体交易函数，简单的交易思路为：
    当策略持多仓时，根据昨天的数据决定继续持仓，平仓还是转仓，如果昨天因子值最大的币对为当前持仓的币对，则继续持仓，
    不做交易。如果昨天因子值最大的币对不是当前持仓币对，但也属于目标交易币对，而且该目标币对的收盘价高于其昨日的25
    日均价时，则选择转仓，也就是以当日开盘价卖出现有持仓再以当日开盘价买入目标币对。如果以上都不是，则平仓。

    当策略空仓时，根据昨天的数据决定开仓做多还是不进行交易。如果昨天因子值最大的币对属于目标币对且其昨日收盘价大于该
    币对昨日的25日均价，则做多，否则不交易。

    因为数据缺失的原因，策略还有一种特殊情况。当要对现有持仓进行平仓或者转仓操作时，如果所要买入或者卖出的币对当日价
    格数据缺失，则跳过该日的买卖操作，将该币对做停牌处理。具体来说，如果要买入开仓时，价格缺失则跳过不买入；如果要卖
    出平仓时数据缺失则跳过不平仓；如果要转仓时，遇到现有持仓币对数据缺失则不卖出并直接进入下一个交易日，如果现有持仓
    币对数据不缺失，但是要转仓时新买入的币对数据缺失则仅卖出之前持仓并空仓状态进入下一个交易日。

    :param df: 用于策略回测的基础数据，在这里，该基础数据包括26个币对的价格数据，各个币对的因子数据和目标币对的25日移动平均价数据
    :param cash_list: 现金账号列表，列表的初始值10000，代表10000个USDT
    :param asset_list: 资产/净值账号列表，净值=现金值+所持有币对数目*该币对当日收盘价
    :param buy_list: 每次交易买入的币对列表
    :param btcnum_list: 买入币对的币对数目列表
    :param n: 用于策略回测数据的行号，n从1开始，因为每一根K线的交易行为由前一根K线的因子值等数据决定，故而n从1开始
    :param close_list: 买入的币对的收盘价列表
    :param max_value_list: 每次迭代时因子值最大的币对的因子数值列表
    :param posittion: 交易所用的仓位大小，取直0或1
    :param win_times: 整个交易的获胜次数，获胜的定义为卖出时的卖出价高于买入时的买入价
    :param price_list: 每次开仓买入时，记录买入时的买入价
    :return: 函数最后返回现金列表，资产/净值账户列表，买入的币对列表，买入的币对数目列表，买入币对该日（根K线）的收盘价列表，因子数值列表，获胜次数
    """
    if n == len(df):
        return cash_list, asset_list, buy_list, btcnum_list, close_list, max_value_list, win_times

    df_last_min = df[symbols].ix[n-1]
    min_idx = df_last_min.idxmax()
    max_value = df_last_min.max()
    max_value_list.append(max_value)
    # print(df_last_min)
    # print(max_value,min_idx)
    df_now_all = df.ix[n]
    # print(df_now_all["date_time"],asset_list[-1],buy_list[-1],min_idx)

    if min_idx in ["btcusdt", "ethusdt", "xrpusdt", "eosusdt", "trxusdt"]:
        if buy_list[-1] == min_idx:
            if df_now_all[min_idx + "_close"]*0 != 0:
                print("数据缺失_c", df_now_all[min_idx + "_close"])
                cash_list.append(cash_list[-1])
                buy_list.append(min_idx)
                asset_list.append(asset_list[-1])
                btcnum_list.append(btcnum_list[-1])
                close_list.append(df_now_all[min_idx + "_close"])

                # logger.info(df.ix[n].date_time)
                # logger.info("继续持仓%s"%min_idx)
                # logger.info("此时的资产总值为%s"%asset_list[-1])
                # logger.info("此时的最大因子值为%s"%max_value)

                return multiple_factor(df, cash_list, asset_list, buy_list, btcnum_list, n + 1, close_list,
                                       max_value_list, posittion, win_times, price_list)
            else:
                cash_list.append(cash_list[-1])
                buy_list.append(min_idx)
                asset_list.append(btcnum_list[-1] * df_now_all[min_idx + "_close"] + cash_list[-1])
                btcnum_list.append(btcnum_list[-1])
                close_list.append(df_now_all[min_idx + "_close"])

                # logger.info(df.ix[n].date_time)
                # logger.info("继续持仓%s"%min_idx)
                # logger.info("此时的资产总值为%s"%asset_list[-1])
                # logger.info("此时的最大因子值为%s"%max_value)

                return multiple_factor(df, cash_list, asset_list, buy_list, btcnum_list, n + 1, close_list,
                                       max_value_list, posittion, win_times, price_list)
        else:
            if type(buy_list[-1]) == str:
                sell_price = df_now_all[buy_list[-1] + "_open"] * (1 - slippage)
                if sell_price*0 != 0:
                    print("数据缺失_z", sell_price)
                    cash_list.append(cash_list[-1])
                    asset_list.append(asset_list[-1])
                    buy_list.append(buy_list[-1])
                    btcnum_list.append(btcnum_list[-1])
                    close_list.append(close_list[-1])
                    # logger.info("数据缺失，不卖%s，不买%s" % (buy_list[-1],min_idx))
                    # logger.info("此时的资产总值为%s" % asset_list[-1])
                    # logger.info("此时的最大因子值为%s" % max_value)
                    return multiple_factor(df, cash_list, asset_list, buy_list, btcnum_list, n + 1, close_list,
                                           max_value_list, posittion, win_times, price_list)
                else:
                    # print("定位",n)
                    # logger.info(df.ix[n].date_time)
                    # logger.info("卖出%s，卖价为%s" % (buy_list[-1], sell_price))
                    sell_amount = btcnum_list[-1]
                    cash_get = sell_price * sell_amount
                    if sell_price > price_list[-1] and sell_amount > 0:
                        win_times += 1
                    # asset_diff=sell_price*sell_amount-asset_chg[-1]
                    # asset_diff=np.sign(asset_diff)
                    # if asset_diff<0:
                    #     posittion = posittion + asset_diff * 0.3
                    # else:
                    #     posittion=posittion+asset_diff*0.2
                    # posittion=max(posittion,0.05)
                    # posittion=min(posittion,1)
                    df_last_all = df.ix[n - 1]
                    # df_close = df[symbols_close].ix[n - 2:n - 1]
                    # df_close_diff = df_close.diff()
                    # df_close_diff = df_close_diff.ix[n - 1].values
                    # print(df_close_diff)
                    # print(len(df_close_diff[df_close_diff > 0]))
                    # print(len(df_close_diff[df_close_diff > -9999999]))
                    # print(len(df_close_diff[df_close_diff>0]))
                    # length_diff = len(df_close_diff[df_close_diff > -9999999])
                    if df_last_all[min_idx+"_close"] > df_last_all[min_idx+"_close25"]:
                        posittion = 1
                    else:
                        posittion = 0
                    buy_price = df_now_all[min_idx + "_open"] * (1 + slippage)
                    if buy_price*0 != 0:
                        cash_list.append(cash_get+cash_list[-1])
                        asset_list.append(cash_list[-1])
                        buy_list.append(np.nan)
                        btcnum_list.append(0)
                        close_list.append(0)
                        # logger.info("买价缺失，不买%s" % min_idx)
                        # logger.info("此时的资产总值为%s" % asset_list[-1])
                        # logger.info("此时的最大因子值为%s" % max_value)
                        return multiple_factor(df, cash_list, asset_list, buy_list, btcnum_list, n + 1, close_list,
                                               max_value_list, posittion, win_times, price_list)
                    else:
                        # logger.info("买入%s,买入价格为%s,买入仓位为%s" % (min_idx, buy_price, posittion))
                        buy_amount = ((cash_get + cash_list[-1]) * posittion) / buy_price
                        cash_now = (cash_get + cash_list[-1]) * (1 - posittion)
                        cash_list.append(cash_now)
                        asset_list.append(buy_amount * df_now_all[min_idx + "_close"] + cash_now)
                        # asset_chg.append(buy_amount*buy_price)
                        # logger.info("此时的资产总值为%s" % asset_list[-1])
                        # logger.info("此时的最大因子值为%s" % max_value)
                        buy_list.append(min_idx)
                        btcnum_list.append(buy_amount)
                        close_list.append(df_now_all[min_idx + "_close"])
                        price_list.append(buy_price)
                        return multiple_factor(df, cash_list, asset_list, buy_list, btcnum_list, n + 1, close_list,
                                               max_value_list, posittion, win_times, price_list)
            else:
                # logger.info(df.ix[n].date_time)
                buy_price = df_now_all[min_idx + "_open"] * (1 + slippage)

                if n > 1:
                    df_last_all = df.ix[n - 1]
                    # df_close = df[symbols_close].ix[n - 2:n - 1]
                    # df_close_diff = df_close.diff()
                    # df_close_diff = df_close_diff.ix[n - 1].values
                    # print(df_close_diff)
                    # print(len(df_close_diff[df_close_diff > 0]))
                    # print(len(df_close_diff[df_close_diff > -9999999]))
                    # print(len(df_close_diff[df_close_diff>0]))
                    # length_diff = len(df_close_diff[df_close_diff > -9999999])
                    if df_last_all[min_idx+"_close"] > df_last_all[min_idx+"_close25"]:
                        posittion = 1
                    else:
                        posittion = 0

                if buy_price*0 != 0:
                    print("数据缺失_k", buy_price)
                    # logger.info("买入数据缺失，不买%s" % min_idx)
                    cash_list.append(cash_list[-1])
                    asset_list.append(asset_list[-1])
                    buy_list.append(buy_list[-1])
                    btcnum_list.append(btcnum_list[-1])
                    close_list.append(close_list[-1])
                    # logger.info("此时的资产总值为%s" % asset_list[-1])
                    # logger.info("此时的最大因子值为%s" % max_value)
                    return multiple_factor(df, cash_list, asset_list, buy_list, btcnum_list, n + 1, close_list,
                                           max_value_list, posittion, win_times, price_list)
                else:
                    # logger.info("开仓买入%s,买入价为%s，买入的仓位为%s" % (min_idx, buy_price, posittion))
                    buy_amount = cash_list[-1]*posittion / buy_price
                    cash_list.append(cash_list[-1]*(1-posittion))
                    asset_list.append(buy_amount * df_now_all[min_idx + "_close"]+cash_list[-1])
                    # logger.info("此时的资产总值为%s" % asset_list[-1])
                    # logger.info("此时的最大因子值为%s" % max_value)
                    # asset_chg.append(buy_price*buy_amount)
                    buy_list.append(min_idx)
                    btcnum_list.append(buy_amount)
                    close_list.append(df_now_all[min_idx + "_close"])
                    price_list.append(buy_price)
                    return multiple_factor(df, cash_list, asset_list, buy_list, btcnum_list, n + 1, close_list,
                                           max_value_list, posittion, win_times, price_list)
    else:
        # logger.info(df.ix[n].date_time)
        if type(buy_list[-1]) == str:
            sell_price = df_now_all[buy_list[-1] + "_open"] * (1 - slippage)
            if sell_price*0 != 0:
                print("数据缺失_p", sell_price)
                # logger.info("数据缺失，不卖出%s" % buy_list[-1])
                cash_list.append(cash_list[-1])
                asset_list.append(asset_list[-1])
                buy_list.append(buy_list[-1])
                btcnum_list.append(btcnum_list[-1])
                close_list.append(close_list[-1])
                # logger.info("此时的资产总值为%s" % asset_list[-1])
                # logger.info("此时的最大因子值为%s" % max_value)
                return multiple_factor(df, cash_list, asset_list, buy_list, btcnum_list, n + 1, close_list,
                                       max_value_list, posittion, win_times, price_list)
            else:
                # logger.info("卖出%s" % buy_list[-1])
                sell_amount = btcnum_list[-1]
                cash_get = sell_price * sell_amount + cash_list[-1]
                cash_list.append(cash_get)
                asset_list.append(cash_get)
                buy_list.append(np.nan)
                btcnum_list.append(0)
                close_list.append(0)
                # logger.info("此时的资产总值为%s" % asset_list[-1])
                # logger.info("此时的最大因子值为%s" % max_value)
                if sell_price > price_list[-1] and sell_amount > 0:
                    win_times += 1
                return multiple_factor(df, cash_list, asset_list, buy_list, btcnum_list, n + 1, close_list,
                                       max_value_list, posittion, win_times, price_list)
        else:
            cash_list.append(cash_list[-1])
            asset_list.append(asset_list[-1])
            buy_list.append(buy_list[-1])
            btcnum_list.append(btcnum_list[-1])
            close_list.append(close_list[-1])
            return multiple_factor(df, cash_list, asset_list, buy_list, btcnum_list, n + 1, close_list, max_value_list,
                                   posittion, win_times, price_list)



def total_ret(net):
    # print(net.iloc[-1],net.iloc[0])
    return net.iloc[-1] / net.iloc[0] - 1


def long_ret(net_df):
    net_df = net_df.copy()
    net_df['ret'] = net_df['net'] / net_df['net'].shift(1)
    long_ret = np.prod(net_df['ret']) - 1
    return long_ret


def month_profit(net_df):
    net_df = net_df.set_index('date_time')
    ret = []
    for gid, group in net_df.groupby(pd.Grouper(freq='M')):
        gid = group.index[0]
        long_return = long_ret(group)
        ret.append([gid, long_return])

    month_ret = pd.DataFrame(ret, columns=['date_time', 'long_return'])
    return month_ret


def annual_ret(net):
    # input daily net
    tot_ret = total_ret(net)
    day = len(net)
    return (tot_ret + 1) ** (365.0 / day) - 1


def AnnualVolatility(net):
    net=net.dropna()
    net_values=net.values
    logreturns = np.diff(np.log(net_values))
    annualVolatility = np.std(logreturns) / np.mean(logreturns)
    annualVolatility = annualVolatility / np.sqrt(1 / 365)
    return annualVolatility


def sharpe_ratio(net):
    # input daily net
    try:
        ret = np.log(net / net.shift(1))
        sharpe = ret.mean() / ret.std() * 365 ** 0.5
    except:
        sharpe = np.nan
    return sharpe


def max_drawdown(A):
    I = -99999999
    for i in range(len(A)-1):
        a = A[i+1:]
        min_a=min(a)
        maxval = 1-min_a/A[i]
        I = max(I,maxval)
    return I*100


def mkfpath(folder, fname):
    try:
        os.mkdir(folder)
    except:
        pass
    fpath = folder + '/' + fname
    return fpath


def summary_net(net_df, plot_in_loops,alpha):
    month_ret = month_profit(net_df)
    # 转换成日净值
    net_df.set_index('date_time', inplace=True)
    # print(net_df)
    # print(len(net_df))
    net_df = net_df.resample(rule='1D').apply({"net":"last","index":"last"})
    # net_df.reset_index(inplace=True)
    # print(net_df.asfreq())
    net_df.dropna(how="all",inplace=True)
    print(net_df)

    # 计算汇总
    net_df["date_time"]=net_df.index
    net = net_df['net']
    tot_ret = total_ret(net)
    ann_ret = annual_ret(net)
    sharpe = sharpe_ratio(net)
    annualVolatility = AnnualVolatility(net)
    drawdown = max_drawdown(net.values)
    ret_r = ann_ret / drawdown

    result = [tot_ret, ann_ret, sharpe, annualVolatility,
              drawdown, ret_r,
              net_df['date_time'].iloc[0], net_df['date_time'].iloc[-1]]
    cols = ['tot_ret', 'ann_ret', 'sharpe', 'annualVolatility', 'max_drawdown', 'ret_ratio', 'start_time', 'end_time']

    if plot_in_loops:
        param_str=alpha

        net_df['index'] = net_df["index"]/net_df["index"].iloc[0]
        net_df['net'] = net_df['net'] / net_df['net'].iloc[0]
        # net_df["index_ma5"]=net_df["index_ma5"].values/10
        # net_df["index_ma20"]=net_df["index_ma20"].values/10

        fpath = mkfpath('/Users/wuyong/alldata/factor_writedb/factor_stra/', param_str + '.png')

        fig, ax = plt.subplots(2)
        net_df.plot(x='date_time', y=['index', 'net'],title=param_str, grid=True, ax=ax[0])
        ax[0].set_xlabel('')
        month_ret['month'] = month_ret['date_time'].dt.strftime('%Y-%m')
        month_ret.plot(kind='bar', x='month',
                       y=['long_return'],
                       color=['r'], grid=True, ax=ax[1])
        plt.tight_layout()
        plt.savefig(fpath)
        # plt.show()

    return result, cols


a=list(range(1, 202))
alpha_test = []
for x in a:
    if x < 10:
        alpha_test.append("Alpha.alpha00"+str(x))
    elif 10 < x < 100:
        alpha_test.append("Alpha.alpha0"+str(x))
    else:
        alpha_test.append("Alpha.alpha" + str(x))

alpha_test = ["Alpha.alpha186"]
print("max")
stat_ls = []
for alpha in alpha_test:
    # 计算出每个alpha的策略指标
    try:
        for symbol in symbols:
            data = pd.read_csv('/Users/wuyong/alldata/factor_writedb/factor_stra/' + symbol + "_" + alpha + "_gtja1h" + '.csv',index_col=0)

            if symbol == "btcusdt":
                df = pd.DataFrame({symbol:data[alpha].values, symbol+"_close":data["close"].values, symbol+"_open":data["open"].values}, index=data["date"].values)
            else:
                df_1 = pd.DataFrame({symbol:data[alpha].values, symbol+"_close":data["close"].values, symbol+"_open":data["open"].values}, index=data["date"].values)
                df = pd.concat([df, df_1], axis=1)

        df["index"] = range(len(df))
        df["date_time"] = df.index
        df.index = df["index"]
        # print(df.head(20))
        print(alpha)
        df[symbols_close] = df[symbols_close].fillna(method="ffill")[symbols_close]
        dataf = read_data("BITMEX", ".bxbt", '1h', "2017-01-01", "2018-10-01")
        index_values = list(dataf["close"].values)
        index_values = [index_values[0]] * 1740 + index_values
        # print(index_values)
        df["index"] = np.array(index_values)
        combine_value = pd.Series(np.zeros(len(df)), name="mid_value")
        for close in ["ethusdt_close", "btcusdt_close", "xrpusdt_close", "eosusdt_close", "trxusdt_close"]:
            df[close + str(25)] = ta.MA(df[close].values, timeperiod=25, matype=0)
            combine_value[df[close] > df[close + str(25)]] = combine_value + 1
        df["strength"] = combine_value
        cash_list, asset_list, buy_list, btcnum_list, close_list, max_value_list, win_times= multiple_factor(df,
                                                                                                   cash_list=[10000],
                                                                                                   asset_list=[10000],
                                                                                                   buy_list=[np.nan],
                                                                                                   btcnum_list=[0], n=1,
                                                                                                   close_list=[0],
                                                                                                   max_value_list=[0])
        df_result = pd.DataFrame({"cash": cash_list, "asset": asset_list, "buy": buy_list,
                                  "coinnum": btcnum_list,"close": close_list,
                                  "max_value": max_value_list}, index=df["date_time"])
        df_result["asset_diff"] = df_result["asset"].diff()
        trade_times = df_result["coinnum"].diff().values
        trade_times = len(np.where(trade_times != 0)[0]) - 1
        df_result["date_time"] = df_result.index
        df_result.index = range(len(df_result))
        # print(df_result[["asset","buy","asset_diff"]])
        # print(win_times)

        # 计算出各个币对上的具体盈亏数额
        sum_ret_symbol = []
        for i in symbols:
            df_mid1 = df_result[df_result["buy"] == i]
            # print(df_mid1)
            # sum_ret_symbol.append(df_mid1[df_mid1["buy"]==i]["asset_diff"].sum())
            index_list = df_mid1.index
            # print(type(index_list))
            # print(index_list)
            # print(index_list+1)
            index_list = list(set(list(index_list) + list(index_list + 1)))
            index_list = sorted(index_list)
            # print(df_result.ix[index_list][["buy","asset_diff"]])
            df_mid2 = df_result.ix[index_list][["buy", "asset_diff"]]
            # print(df_mid2.dropna(how="all"))
            df_mid2.dropna(how="all", inplace=True)
            # print(df_mid2)
            df_mid2.fillna("x", inplace=True)
            df_mid2 = df_mid2[(df_mid2["buy"] == i) | (df_mid2["buy"] == "x")]
            sum_ret_symbol.append(df_mid2["asset_diff"].sum())
            # exit()

        df_result["net"] = df_result["asset"]
        # print(df_result)
        df_result["index"] = df["index"]
        # df_result["index_ma5"]=df["index_ma5"]
        # df_result["index_ma20"]=df["index_ma20"]
        df_result["date_time"] = pd.to_datetime(df_result["date_time"])
        # print(df_result[["net","asset_diff","buy","asset","date_time"]])
        result, cols = summary_net(df_result[["net", "close", "index", "date_time"]], 1, alpha + "_bitfinex_positive_1h")
        result = result + sum_ret_symbol
        result = [trade_times, alpha, win_times] + result
        cols = cols + symbols
        cols = ["trade_times", "alpha", "win_times"] + cols
        stat_ls.append(result)
        # df_last=pd.DataFrame(stat_ls, columns=cols)
    except (FileNotFoundError) as e:
        pass

df_last=pd.DataFrame(stat_ls,columns=cols)
print(df_last)
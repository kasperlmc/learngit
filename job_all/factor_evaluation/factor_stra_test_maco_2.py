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
def multiple_factor(df,cash_list=[10000],asset_list=[10000],buy_list=[np.nan],btcnum_list=[0],n=1,close_list=[0],max_value_list=[0],posittion=1):
    if n==len(df):
        return cash_list,asset_list,buy_list,btcnum_list,close_list,max_value_list

    df_last_min=df[symbols].ix[n-1]
    min_idx=df_last_min.idxmax()
    max_value=df_last_min.max()
    max_value_list.append(max_value)
    df_now_all = df.ix[n]

    if min_idx in ["btcusdt", "ethusdt", "xrpusdt", "eosusdt", "trxusdt"]:
        if buy_list[-1] == min_idx:
            if df_now_all[min_idx + "_close"]*0 != 0:
                print("数据缺失_c",df_now_all[min_idx + "_close"])
                cash_list.append(cash_list[-1])
                buy_list.append(min_idx)
                asset_list.append(asset_list[-1])
                btcnum_list.append(btcnum_list[-1])
                close_list.append(df_now_all[min_idx + "_close"])
                return multiple_factor(df, cash_list, asset_list, buy_list, btcnum_list, n + 1, close_list,
                                       max_value_list,posittion)
            else:
                cash_list.append(cash_list[-1])
                buy_list.append(min_idx)
                asset_list.append(btcnum_list[-1] * df_now_all[min_idx + "_close"] + cash_list[-1])
                btcnum_list.append(btcnum_list[-1])
                close_list.append(df_now_all[min_idx + "_close"])
                return multiple_factor(df, cash_list, asset_list, buy_list, btcnum_list, n + 1, close_list,
                                       max_value_list,posittion)
        else:
            if type(buy_list[-1]) == str:
                sell_price = df_now_all[buy_list[-1] + "_open"] * (1 - slippage)
                if sell_price*0 != 0:
                    print("数据缺失_z",sell_price)
                    cash_list.append(cash_list[-1])
                    asset_list.append(asset_list[-1])
                    buy_list.append(buy_list[-1])
                    btcnum_list.append(btcnum_list[-1])
                    close_list.append(close_list[-1])
                    return multiple_factor(df, cash_list, asset_list, buy_list, btcnum_list, n + 1, close_list,
                                           max_value_list,posittion)
                else:
                    sell_amount = btcnum_list[-1]
                    cash_get = sell_price * sell_amount
                    # df_close = df[symbols_close].ix[n - 2:n - 1]
                    # df_close_diff = df_close.diff()
                    # df_close_diff = df_close_diff.ix[n - 1].values
                    # length_diff = len(df_close_diff[df_close_diff > 0]) /
                    # len(df_close_diff[df_close_diff > -9999999]) - 0.5
                    # length_diff = np.sign(length_diff)
                    # position = max(length_diff, 0)
                    # strength_value = df_now_all[min_idx+"_closecom"]-5
                    # strength_value = np.sign(strength_value)
                    # posittion = max(strength_value,0)
                    df_last_all = df.ix[n - 1]
                    strength_value = df_last_all[min_idx + "_closecom"]
                    if strength_value < 3:
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
                        return multiple_factor(df, cash_list, asset_list, buy_list, btcnum_list, n + 1, close_list,
                                               max_value_list, posittion)
                    else:
                        buy_amount = ((cash_get + cash_list[-1]) * posittion) / buy_price
                        cash_now = (cash_get + cash_list[-1]) * (1 - posittion)
                        cash_list.append(cash_now)
                        asset_list.append(buy_amount * df_now_all[min_idx + "_close"] + cash_now)
                        buy_list.append(min_idx)
                        btcnum_list.append(buy_amount)
                        close_list.append(df_now_all[min_idx + "_close"])
                        return multiple_factor(df, cash_list, asset_list, buy_list, btcnum_list, n + 1, close_list,
                                               max_value_list, posittion)
            else:
                # logger.info(df.ix[n].date_time)
                buy_price = df_now_all[min_idx + "_open"] * (1 + slippage)
                if buy_price*0 != 0:
                    print("数据缺失_k",buy_price)
                    cash_list.append(cash_list[-1])
                    asset_list.append(asset_list[-1])
                    buy_list.append(buy_list[-1])
                    btcnum_list.append(btcnum_list[-1])
                    close_list.append(close_list[-1])
                    return multiple_factor(df, cash_list, asset_list, buy_list, btcnum_list, n + 1, close_list,
                                           max_value_list,posittion)
                else:
                    buy_amount = cash_list[-1] / buy_price
                    cash_list.append(0)
                    asset_list.append(buy_amount * df_now_all[min_idx + "_close"])
                    # asset_chg.append(buy_price*buy_amount)
                    buy_list.append(min_idx)
                    btcnum_list.append(buy_amount)
                    close_list.append(df_now_all[min_idx + "_close"])
                    return multiple_factor(df, cash_list, asset_list, buy_list, btcnum_list, n + 1, close_list,
                                           max_value_list, posittion)
    else:
        if type(buy_list[-1]) == str:
            sell_price = df_now_all[buy_list[-1] + "_open"] * (1 - slippage)
            if sell_price*0 != 0:
                print("数据缺失_p",sell_price)
                cash_list.append(cash_list[-1])
                asset_list.append(asset_list[-1])
                buy_list.append(buy_list[-1])
                btcnum_list.append(btcnum_list[-1])
                close_list.append(close_list[-1])
                return multiple_factor(df, cash_list, asset_list, buy_list, btcnum_list, n + 1, close_list,
                                       max_value_list,
                                       posittion)
            else:
                sell_amount = btcnum_list[-1]
                cash_get = sell_price * sell_amount + cash_list[-1]
                cash_list.append(cash_get)
                asset_list.append(cash_get)
                buy_list.append(np.nan)
                btcnum_list.append(0)
                close_list.append(0)
                return multiple_factor(df, cash_list, asset_list, buy_list, btcnum_list, n + 1, close_list,
                                       max_value_list,posittion)
        else:
            cash_list.append(cash_list[-1])
            asset_list.append(asset_list[-1])
            buy_list.append(buy_list[-1])
            btcnum_list.append(btcnum_list[-1])
            close_list.append(close_list[-1])
            return multiple_factor(df, cash_list, asset_list, buy_list, btcnum_list, n + 1, close_list, max_value_list,
                                   posittion)


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

# alpha_test = ["Alpha.alpha020"]
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
        ma_list = [3, 5, 10, 15, 30, 60, 120, 240]
        for close in symbols_close:
            # df[close+"com"] = np.zeros(len(df))
            combine_value = pd.Series(np.zeros(len(df)),name="mid_value")
            for i in range(len(ma_list)):
                df[close+str(ma_list[i])] = ta.MA(df[close].values, timeperiod=ma_list[i], matype=0)
                if i==0:
                    combine_value[df[close]>df[close+str(ma_list[i])]]=combine_value+1
                else:
                    combine_value[df[close+str(ma_list[i-1])]>df[close+str(ma_list[i])]]=combine_value+1
            df[close+"com"]=combine_value
        # print(df.head(20))

        dataf = read_data("BITMEX", ".bxbt", '1h', "2017-01-01", "2018-10-01")
        index_values = list(dataf["close"].values)
        index_values = [index_values[0]] * 1740 + index_values
        # print(index_values)
        df["index"] = np.array(index_values)
        cash_list, asset_list, buy_list, btcnum_list, close_list, max_value_list = multiple_factor(df,
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
        # print(df_result.tail())

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
        result, cols = summary_net(df_result[["net", "close", "index", "date_time"]], 0, alpha + "_bitfinex_positive")
        result = result + sum_ret_symbol
        result = [trade_times, alpha] + result
        cols = cols + symbols
        cols = ["trade_times", "alpha"] + cols
        stat_ls.append(result)
        # df_last=pd.DataFrame(stat_ls, columns=cols)
    except (FileNotFoundError) as e:
        pass

df_last=pd.DataFrame(stat_ls,columns=cols)
print(df_last)
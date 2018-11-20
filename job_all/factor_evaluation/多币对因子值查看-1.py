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
from sklearn import preprocessing
from lib.factors_gtja import *
import copy

#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)

slippage = 0.00025
@tail_call_optimized
def multiple_factor(df,cash_list=[10000],asset_list=[10000],buy_list=[np.nan],btcnum_list=[0],n=0,close_list=[0],max_value_list=[0]):
    if n==len(df)-1:
        return cash_list,asset_list,buy_list,btcnum_list,close_list,max_value_list

    df_temp_min=df[symbols].ix[n]
    min_idx=df_temp_min.idxmin()
    max_value=df_temp_min.min()
    max_value_list.append(max_value)
    if True:
        if buy_list[-1]==min_idx:
            df_next_all = df.ix[n + 1]
            cash_list.append(cash_list[-1])
            buy_list.append(min_idx)
            asset_list.append(btcnum_list[-1]*df_next_all[min_idx+"_close"])
            btcnum_list.append(btcnum_list[-1])
            close_list.append(df_next_all[min_idx+"_close"])
            return multiple_factor(df,cash_list,asset_list,buy_list,btcnum_list,n+1,close_list,max_value_list)
        else:
            if type(buy_list[-1])==str:
                df_next_all=df.ix[n+1]
                sell_price=df_next_all[buy_list[-1]+"_open"]*(1-slippage)
                sell_amount=btcnum_list[-1]
                cash_get=sell_price*sell_amount+cash_list[-1]
                buy_price=df_next_all[min_idx+"_open"]*(1+slippage)
                buy_amount=cash_get/buy_price
                cash_list.append(0)
                asset_list.append(buy_amount*df_next_all[min_idx+"_close"])
                buy_list.append(min_idx)
                btcnum_list.append(buy_amount)
                close_list.append(df_next_all[min_idx + "_close"])
                return multiple_factor(df,cash_list,asset_list,buy_list,btcnum_list,n+1,close_list,max_value_list)

            else:
                df_next_all = df.ix[n + 1]
                buy_price=df_next_all[min_idx+"_open"]*(1+slippage)
                buy_amount = cash_list[-1] / buy_price
                cash_list.append(0)
                asset_list.append(buy_amount * df_next_all[min_idx + "_close"])
                buy_list.append(min_idx)
                btcnum_list.append(buy_amount)
                close_list.append(df_next_all[min_idx + "_close"])
                return multiple_factor(df,cash_list, asset_list, buy_list, btcnum_list, n + 1,close_list,max_value_list)
    else:
        if type(buy_list[-1]) == str:
            df_next_all = df.ix[n + 1]
            sell_price = df_next_all[buy_list[-1] + "_open"] * (1 - slippage)
            sell_amount = btcnum_list[-1]
            cash_get = sell_price * sell_amount + cash_list[-1]
            cash_list.append(cash_get)
            asset_list.append(cash_get)
            buy_list.append(np.nan)
            btcnum_list.append(0)
            close_list.append(0)
            return multiple_factor(df,cash_list, asset_list, buy_list, btcnum_list, n + 1,close_list,max_value_list)
        else:
            cash_list.append(cash_list[-1])
            asset_list.append(asset_list[-1])
            buy_list.append(buy_list[-1])
            btcnum_list.append(btcnum_list[-1])
            close_list.append(close_list[-1])
            return multiple_factor(df,cash_list, asset_list, buy_list, btcnum_list, n + 1,close_list,max_value_list)


def total_ret(net):
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


def annual_ret(date_time, net):
    # input daily net
    tot_ret = total_ret(net)
    time_delta = date_time.iloc[-1] - date_time.iloc[0]
    day = time_delta.days
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


def infomation_ratio(date_time, base, net):
    # input daily net
    net_ann_ret = annual_ret(date_time, net)
    base_ann_ret = annual_ret(date_time, base)
    diff_ret_std = (net.pct_change()-base.pct_change()).std() * np.sqrt(365)
    ir = (net_ann_ret - base_ann_ret) / diff_ret_std
    return ir


def max_drawdown(A):
    I = -99999999
    for i in range(len(A)-1):
        a = A[i+1:]
        min_a=min(a)
        maxval = 1-min_a/A[i]
        I = max(I,maxval)
    return I*100

def alpha_beta(date_time, base, net):
    # beta 是日收益的线性拟合斜率
    base_p = base.pct_change()
    net_p = net.pct_change()
    days = len(base_p) - 1
    beta = (days * (base_p * net_p).sum() - base_p.sum() * net_p.sum()
            ) / (days * (base_p * base_p).sum() - base_p.sum() * base_p.sum())

    # 3%无风险收益
    rf = 0.03
    alpha = (annual_ret(date_time, net) - rf
             - beta * (annual_ret(date_time, base) - rf))
    return alpha, beta

def mkfpath(folder, fname):
    try:
        os.mkdir(folder)
    except:
        pass
    fpath = folder + '\\' + fname
    return fpath

def summary_net(net_df, plot_in_loops,alphas):
    month_ret = month_profit(net_df)
    # 转换成日净值
    net_df.set_index('date_time', inplace=True)
    net_df = net_df.resample('1D').asfreq()
    net_df.reset_index(inplace=True)
    net_df.dropna(inplace=True)

    # 计算汇总
    net = net_df['net']
    date_time = net_df['date_time']
    base = net_df['close']
    tot_ret = total_ret(net)
    ann_ret = annual_ret(date_time, net)
    sharpe = sharpe_ratio(net)
    annualVolatility = AnnualVolatility(net)
    drawdown = max_drawdown(net.values)
    alpha, beta = alpha_beta(date_time, base, net)
    ir = infomation_ratio(date_time, base, net)
    ret_r = ann_ret / drawdown

    result = [tot_ret, ann_ret, sharpe, annualVolatility,
              drawdown, alpha, beta, ret_r, ir,
              net_df['date_time'].iloc[0], net_df['date_time'].iloc[-1]]
    cols = ['tot_ret', 'ann_ret', 'sharpe', 'annualVolatility', 'max_drawdown', 'alpha', 'beta', 'ret_ratio', 'ir', 'start_time', 'end_time']

    if plot_in_loops:
        param_str="multiple"+alphas

        net_df['index'] =net_df["index"]/net_df["index"].iloc[0]
        net_df['net'] = net_df['net'] / net_df['net'].iloc[0]

        fpath = mkfpath('api_figure_1', param_str + '.png')

        fig, ax = plt.subplots(2)
        net_df.plot(x='date_time', y=['index', 'net'],title=param_str, grid=True, ax=ax[0])
        ax[0].set_xlabel('')
        month_ret['month'] = month_ret['date_time'].dt.strftime('%Y-%m')
        month_ret.plot(kind='bar', x='month',
                       y=['long_return'],
                       color=['r'], grid=True, ax=ax[1])
        plt.tight_layout()
        plt.savefig(fpath)

    return result, cols

symbols = ['btcusdt',"ethusdt","xrpusdt","zecusdt","eosusdt","neousdt","ltcusdt","etcusdt","etpusdt","iotusdt"]

'''
alpha_test=["Alpha.alpha003","Alpha.alpha007","Alpha.alpha009","Alpha.alpha014","Alpha.alpha018","Alpha.alpha019",
            "Alpha.alpha020","Alpha.alpha024","Alpha.alpha028","Alpha.alpha031","Alpha.alpha037","Alpha.alpha038",
            "Alpha.alpha040","Alpha.alpha050","Alpha.alpha051","Alpha.alpha052","Alpha.alpha053","Alpha.alpha057",
            "Alpha.alpha063","Alpha.alpha065","Alpha.alpha066","Alpha.alpha067","Alpha.alpha069","Alpha.alpha070",
            "Alpha.alpha071","Alpha.alpha079","Alpha.alpha084","Alpha.alpha088","Alpha.alpha093","Alpha.alpha095",
            "Alpha.alpha096","Alpha.alpha106","Alpha.alpha110","Alpha.alpha118","Alpha.alpha126","Alpha.alpha128",
            "Alpha.alpha133","Alpha.alpha134","Alpha.alpha152","Alpha.alpha161","Alpha.alpha162","Alpha.alpha167",
            "Alpha.alpha172","Alpha.alpha177","Alpha.alpha186","Alpha.alpha189"]
'''

a=list(range(1,202))
alpha_test=[]
for x in a:
    if x<10:
        alpha_test.append("Alpha.alpha00"+str(x))
    elif 10<x<100:
        alpha_test.append("Alpha.alpha0"+str(x))
    else:
        alpha_test.append("Alpha.alpha" + str(x))

alpha_test=["Alpha.alpha018"]
for alpha in alpha_test:
    try:
        for symbol in symbols:
            data=pd.read_csv('../factor_writedb/multiple_subject/' + symbol + "_" + alpha +"_gtja1h" + '.csv',index_col=0)

            if symbol=="btcusdt":
                df=pd.DataFrame({symbol:data[alpha].values,symbol+"_close":data["close"].values,symbol+"_open":data["open"].values},index=data["date"].values)
            else:
                df_1=pd.DataFrame({symbol:data[alpha].values,symbol+"_close":data["close"].values,symbol+"_open":data["open"].values},index=data["date"].values)
                df=pd.concat([df,df_1],axis=1)

        df.dropna(inplace=True)
        #print(df.ix["2018-03-01 00:00:00":])
        #df=df.ix["2018-03-01 00:00:00":]
        df["index"]=range(len(df))
        df["date_time"]=df.index
        df.index=df["index"]
        print(alpha)
        symbols_close=[x+"_close" for x in symbols]
        df["index"]=(df[symbols_close]/df[symbols_close].iloc[0]).sum(axis=1)
        cash_list,asset_list,buy_list,btcnum_list,close_list,max_value_list=multiple_factor(df,cash_list=[10000],asset_list=[10000],buy_list=[np.nan],btcnum_list=[0],n=0,close_list=[0],max_value_list=[0])
        df_result=pd.DataFrame({"cash":cash_list,"asset":asset_list,"buy":buy_list,"coinnum":btcnum_list,"close":close_list,"max_value":max_value_list},index=df["date_time"])
        df_result["asset_diff"]=df_result["asset"].diff()
        sum_ret_symbol=[]
        for i in symbols:
            df_mid1=df_result[["asset_diff","buy"]]
            sum_ret_symbol.append(df_mid1[df_mid1["buy"]==i]["asset_diff"].sum())

        trade_times=df_result["coinnum"].diff().values
        trade_times=len(np.where(trade_times!=0)[0])-1
        df_result["date_time"]=df_result.index
        df_result.index=range(len(df_result))
        df_result["net"]=df_result["asset"]
        #print(df_result)
        df_result["index"]=df["index"]
        df_result["date_time"] = pd.to_datetime(df_result["date_time"])
        result,cols=summary_net(df_result,1,alpha+"_bitfinex_negetive")
        result=result+sum_ret_symbol
        result.append(trade_times)
        cols=cols+symbols
        cols.append("trade_times")
        stat_ls=[result]
        df_last=pd.DataFrame(stat_ls,columns=cols)
        print(df_last)
    except :
        print(alpha)























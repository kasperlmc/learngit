# coding=utf-8

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
import talib
import time



def mkfpath(folder, fname):
    try:
        os.mkdir(folder)
    except:
        pass
    fpath = folder + '\\' + fname
    return fpath


def alarm(n):
    for _ in range(n):
        os.system('afplay /System/Library/Sounds/Hero.aiff')


def total_ret(net):
    return net.iloc[-1] / net.iloc[0] - 1


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


def mean_position(pos):
    return pos.mean()


def mean_hold_k(pos):
    # 开仓记录持有期
    pos_values=pos.values

    long_hold=np.where(pos_values>0)[0]
    hold_diff_l=np.diff(long_hold)
    long_hold_average=len(long_hold)/(len(hold_diff_l[hold_diff_l>1])+1)

    short_hold = np.where(pos_values < 0)[0]
    hold_diff_s = np.diff(short_hold)
    short_hold_average = len(short_hold) / (len(hold_diff_s[hold_diff_s > 1])+1)

    return long_hold_average,short_hold_average


def trade_times(pos):
    # 平仓记录交易次数
    pos_values = pos.values

    if pos_values[-1] > 0:
        long_hold = np.where(pos_values > 0)[0]
        hold_diff_l = np.diff(long_hold)
        long_times=len(hold_diff_l[hold_diff_l>1])
        short_hold = np.where(pos_values < 0)[0]
        hold_diff_s = np.diff(short_hold)
        short_times=len(hold_diff_s[hold_diff_s>1])+1
    elif pos_values[-1]==0:
        long_hold = np.where(pos_values > 0)[0]
        hold_diff_l = np.diff(long_hold)
        long_times = len(hold_diff_l[hold_diff_l > 1]) + 1
        short_hold = np.where(pos_values < 0)[0]
        hold_diff_s = np.diff(short_hold)
        short_times = len(hold_diff_s[hold_diff_s > 1]) + 1
    else:
        long_hold = np.where(pos_values > 0)[0]
        hold_diff_l = np.diff(long_hold)
        long_times = len(hold_diff_l[hold_diff_l > 1]) + 1
        short_hold = np.where(pos_values < 0)[0]
        hold_diff_s = np.diff(short_hold)
        short_times = len(hold_diff_s[hold_diff_s > 1])

    return long_times, short_times


def win_profit_ratio(signal_df):
    # 胜率和盈亏比，当前脚本只适用于全仓进出，待完善
    buy = signal_df[signal_df['signal'].str.startswith('b')]
    buy_price = np.array(buy['price'])
    sell = signal_df[signal_df['signal'].str.startswith('s')]
    sell_price = np.array(sell['price'])
    if len(buy_price) > len(sell_price):
        buy_price = buy_price[:-1]
    elif len(buy_price) < len(sell_price):
        sell_price = sell_price[:-1]

    try:  
        profit = sell_price / buy_price - 1
        win_trade = profit[profit > 0]
        loss_trade = profit[profit <= 0]
        win_ratio = len(win_trade) / len(profit)
        profit_ratio = - np.mean(win_trade) / np.mean(loss_trade)
    except:
        win_ratio = -1
        profit_ratio = -1

    return win_ratio, profit_ratio


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


def resample(df, rule):
    return df.resample(rule, on='date_time').apply({'open': 'first',
                                                    'high': 'max',
                                                    'low': 'min',
                                                    'close': 'last',
                                                    'volume': 'sum'})


def order_pct_to(pct, capital, price, fee):
    now_pct = capital[0] * price / (capital[0] * price + capital[1])
    # capital : [s0, s1] -> [e0, e1]
    [s0, s1] = capital
    if now_pct < pct:
        # 需要买, 按实际成交额交手续费
        # delta = (e0 - s0) * price
        # e0 * price + e1 = s0 * price + s1 - delta * fee
        # e0 * price / (e0 * price + e1) = pct
        e0 = pct*(fee*price*s0 + price*s0 + s1)/(price*(fee*pct + 1))
        e1 = (fee*price*s0 + price*s0 + s1) * (1 - pct)/(fee*pct + 1)
        delta = (pct*(price*s0 + s1) - price*s0)/(fee*pct + 1)
    else:
        # 需要卖, 按实际成交额交手续费
        # delta = (e1 - s1)
        # e0 * price + e1 = s0 * price + s1 - delta * fee
        # e0 * price / (e0 * price + e1) = pct
        e0 = -pct*(fee*s1 + price*s0 + s1)/(price*(fee*(pct - 1) - 1))
        e1 = (pct - 1)*(fee*s1 + price*s0 + s1)/(fee*(pct - 1) - 1)
        delta = (s1 + (pct - 1)*(price*s0 + s1))/(fee*(pct - 1) - 1)

    capital = [e0, e1]
    return capital, delta

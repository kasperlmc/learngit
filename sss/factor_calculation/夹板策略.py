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

import os
sys.path.append('..')
from lib.myfun import *
import copy
import pandas as pd
import numpy as np
from sklearn.model_selection import ParameterGrid
import talib as ta
import copy
import logging
import matplotlib.pyplot as plt


#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)

exchange = 'BIAN'
symbols = ['btcusdt']

dataf_h = read_data(exchange, symbols[0], '1h', "2017-01-01", "2018-10-01")
dataf_h.index=dataf_h["date"]

dataf_m = read_data(exchange, symbols[0], '1m', "2017-01-01", "2018-10-01")
dataf_m.index=dataf_m["date"]

def random_wave_m(cash, btcnum, time, n, slippage, x, y, price_basic):
    open_long=price_basic*x
    close_long=price_basic*x*(1-y)
    open_short=price_basic/x
    close_short=price_basic/(x*(1-y))
    data_temp = dataf_m.ix[time]
    length=len(data_temp)
    if n==length-1:
        return cash, btcnum, price_basic

    if btcnum>0:
        if data_temp.ix[n].close<close_long:
            #平多仓
            logger.info(str(data_temp.ix[n]["date"])+"止损多仓")
            logger.info("收盘价%s,平多价%s" % (data_temp.ix[n].close, close_long))
            sell_price=data_temp.ix[n+1].open*(1-slippage)
            cash=cash+sell_price*btcnum
            btcnum=0
            price_basic=sell_price
            return cash,btcnum,price_basic

        elif data_temp.ix[n].close>open_short:
            #转空仓
            logger.info(str(data_temp.ix[n]["date"])+"转空仓")
            logger.info("收盘价%s,开空价%s" % (data_temp.ix[n].close, open_short))
            sell_price = data_temp.ix[n + 1].open * (1 - slippage)
            cash = cash + sell_price * btcnum
            sellamount = cash / (2 * sell_price)
            cash=cash+sellamount*sell_price
            btcnum=-sellamount
            return random_wave_m(cash,btcnum,time,n+1,slippage,x,y,price_basic)

        else:
            return random_wave_m(cash,btcnum,time,n+1,slippage,x,y,price_basic)

    elif btcnum<0:
        if data_temp.ix[n].close>close_short:
            #平空仓
            logger.info(str(data_temp.ix[n]["date"])+"止损空仓")
            logger.info("收盘价%s,平空价%s" % (data_temp.ix[n].close, close_short))
            buy_price=data_temp.ix[n+1].open*(1+slippage)
            cash=cash-buy_price*np.abs(btcnum)
            btcnum=0
            price_basic=buy_price
            return cash,btcnum,price_basic

        elif data_temp.ix[n].close<open_long:
            #转多仓
            logger.info(str(data_temp.ix[n]["date"])+"转多仓")
            logger.info("收盘价%s,开多价%s" % (data_temp.ix[n].close, open_long))
            buy_price=data_temp.ix[n+1].open*(1+slippage)
            cash=cash-buy_price*np.abs(btcnum)
            buyamount=cash/buy_price
            cash=0
            btcnum=buyamount
            return random_wave_m(cash,btcnum,time,n+1,slippage,x,y,price_basic)

        else:
            return random_wave_m(cash,btcnum,time,n+1,slippage,x,y,price_basic)

    else :
        if data_temp.ix[n].close<open_long:
            #做多仓
            logger.info(str(data_temp.ix[n]["date"])+"做多仓")
            logger.info("收盘价%s,开多价%s" % (data_temp.ix[n].close, open_long))
            buy_price = data_temp.ix[n + 1].open * (1 + slippage)
            buyamount=cash/buy_price
            btcnum=buyamount
            cash=0
            return random_wave_m(cash,btcnum,time,n+1,slippage,x,y,price_basic)
        elif data_temp.ix[n].close>open_short:
            #做空仓
            logger.info(str(data_temp.ix[n]["date"])+"做空仓")
            logger.info("收盘价%s,开空价%s"%(data_temp.ix[n].close,open_short))
            sell_price=data_temp.ix[n+1].open*(1-slippage)
            sellamount=cash/(2*sell_price)
            cash=cash+sellamount*sell_price
            btcnum=-sellamount
            return random_wave_m(cash,btcnum,time,n+1,slippage,x,y,price_basic)
        else:
            return random_wave_m(cash,btcnum,time,n+1,slippage,x,y,price_basic)




@tail_call_optimized
def random_wave_h(x, y, cash_list, btcnum_list,price_basic,slippage):

    if len(cash_list)==9730:
        return cash_list,btcnum_list

    data_current = dataf_h.ix[len(cash_list)]
    date_h = str(data_current.date)
    time = date_h[:date_h.find(":")]

    cash=cash_list[-1]
    btcnum=btcnum_list[-1]
    cash,btcnum,price_basic=random_wave_m(cash,btcnum,time,0,slippage,x,y,price_basic)
    cash_list.append(cash)
    btcnum_list.append(btcnum)
    return random_wave_h(x,y,cash_list,btcnum_list,price_basic,slippage)


#print(random_wave_h(0.9,0.025,[10000],[0],4308.83,0.00025))


param_grid={"x":[0.8,0.85,0.9,0.95],"y":[0.015,0.02,0.05,0.075]}
param_list=list(ParameterGrid(param_grid))

print(param_list)

for x in range(len(param_list)):
    param=param_list[x]
    print(x)
    logger = logging.getLogger(str(x))  # logging对象
    fh = logging.FileHandler(str(x) + "jiaban.log")  # 文件对象
    sh = logging.StreamHandler()  # 输出流对象
    fm = logging.Formatter('%(asctime)s-%(filename)s[line%(lineno)d]-%(levelname)s-%(message)s')  # 格式化对象
    fh.setFormatter(fm)  # 设置格式
    sh.setFormatter(fm)  # 设置格式
    logger.addHandler(fh)  # logger添加文件输出流
    logger.addHandler(sh)  # logger添加标准输出流（std out）
    logger.setLevel(logging.INFO)  # 设置从那个等级开始提示
    cash_list,btcnum_list=random_wave_h(param["x"],param["y"],cash_list=[10000],btcnum_list=[0],price_basic=4308.83, slippage=0.00025)
    df = pd.DataFrame({"cash": cash_list, "btc": btcnum_list})
    df.to_csv(str(x) + "_" + "jiaban.csv")











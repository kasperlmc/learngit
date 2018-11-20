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
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"


#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)

exchange = 'BIAN'
symbols = ['btcusdt']

dataf_h = read_data(exchange, symbols[0], '1h', "2017-01-01", "2018-10-01")
dataf_h.index=dataf_h["date"]
#print(dataf_h.head(40))
print(len(dataf_h))
dataf_m = read_data(exchange, symbols[0], '1m', "2017-01-01", "2018-10-01")
dataf_m.index=dataf_m["date"]


def random_wave_m(buy_price, stop_loss, stop_profit, cash, btcnum, time, n, length, slippage, status, price_stop):
    data_temp=dataf_m.ix[time]
    if n==length-1:
        status=0
        return cash,btcnum,status,price_stop,buy_price

    if btcnum > 0:
        if data_temp.ix[n].close>stop_profit:
            logger.info(data_temp.ix[n]["date"])
            sell_price=data_temp.ix[n+1].open*(1-slippage)
            logger.info("止盈卖出:%s"%sell_price)
            cash=cash+sell_price*btcnum
            btcnum=0
            status=1
            return cash,btcnum,status,0,0
        elif data_temp.ix[n].close<stop_loss:
            logger.info(data_temp.ix[n]["date"])
            #print(n)
            sell_price=data_temp.ix[n+1].open*(1-slippage)
            logger.info("止损卖出:%s"%sell_price)
            cash=cash+sell_price*btcnum
            btcnum=0
            status=-1
            return cash,btcnum,status,sell_price,0
        else:
            return random_wave_m(buy_price,stop_loss,stop_profit,cash,btcnum,time,n+1,length,slippage,status,price_stop)

    elif data_temp.ix[n].low < buy_price:
        logger.info(data_temp.ix[n]["date"])
        buy_price=buy_price*(1+slippage)
        logger.info("止损:%s,止盈:%s"%(stop_loss,stop_profit))
        logger.info("买价:%s"%buy_price)
        btcnum=cash/buy_price
        cash=0
        return random_wave_m(buy_price,stop_loss,stop_profit,cash,btcnum,time,n+1,length,slippage,status,price_stop)

    else:
        return random_wave_m(buy_price,stop_loss,stop_profit,cash,btcnum,time,n+1,length,slippage,status,price_stop)


@tail_call_optimized
def random_wave_h(N1, N2, N3, N4, cash_list, btcnum_list,status,price_stop,buy_price,max_price):

    if len(cash_list)==9730:
        return cash_list,btcnum_list

    if btcnum_list[-1]==0:
        if status == 1:
            price = dataf_h["high"].values[len(cash_list) - 1:len(cash_list) + N1 - 1].max()
            max_price=price
            buy_price = price * N2
            stop_loss = price * N3
            stop_profit = price * N4
        elif status==-1:
            price=price_stop
            max_price=price
            buy_price = price * N2
            stop_loss = price * N3
            stop_profit = price * N4

        else:
            price = dataf_h["high"].values[len(cash_list) - 1:len(cash_list) + N1 - 1].max()
            if price>max_price:
                max_price=price
                buy_price = price * N2
                stop_loss = price * N3
                stop_profit = price * N4
            else:
                price=max_price
                buy_price = price * N2
                stop_loss = price * N3
                stop_profit = price * N4

        data_current=dataf_h.ix[len(cash_list) + N1 - 1]
        date_h=str(data_current.date)
        time=date_h[:date_h.find(":")]
        cash=cash_list[-1]
        btcnum=btcnum_list[-1]
        length=len(dataf_m.ix[time])
        slippage = 0.00025
        cash,btcnum,status,price_stop,buy_price=random_wave_m(buy_price,stop_loss,stop_profit,cash,btcnum,time,0,length,slippage,status,price_stop)
        cash_list.append(cash)
        btcnum_list.append(btcnum)
        return random_wave_h(N1, N2, N3, N4, cash_list, btcnum_list,status,price_stop,buy_price,max_price)


    if btcnum_list[-1]>0:
        data_current = dataf_h.ix[len(cash_list) + N1 - 1]
        date_h = str(data_current.date)
        time = date_h[:date_h.find(":")]
        cash = cash_list[-1]
        btcnum = btcnum_list[-1]
        length = len(dataf_m.ix[time])
        slippage = 0.00025
        stop_loss=buy_price/N2*N3
        stop_profit=buy_price/N2*N4
        cash, btcnum, status, price_stop, buy_price = random_wave_m(buy_price, stop_loss, stop_profit, cash, btcnum,
                                                                        time, 0, length, slippage, status,price_stop)
        cash_list.append(cash)
        btcnum_list.append(btcnum)
        return random_wave_h(N1, N2, N3, N4, cash_list, btcnum_list, status, price_stop, buy_price,max_price)


'''
time=s="2018-09-30 23"
buy_price=6650.00
stop_loss=buy_price*(1-0.001)
stop_profit=buy_price*(1+0.05)

cash=10000
btcnum=0
n=0
length=60
slippage=0.00025
'''


#print(random_wave_m(buy_price,stop_loss,stop_profit,cash,btcnum,time,n,length,slippage,1))
#print(random_wave_h(N1=15,N2=0.95,N3=0.925,N4=1.025,cash_list=[10000],btcnum_list=[0],status=1,price_stop=0,buy_price=0))

#cash_list,btcnum_list=random_wave_h(N1=15,N2=0.95,N3=0.925,N4=1.025,cash_list=[10000],btcnum_list=[0],status=1,price_stop=0,buy_price=0)

#df=pd.DataFrame({"cash":cash_list,"btc":btcnum_list})
#print(df)
#df.to_csv("random.csv")

param_grid={"N1":[10,15,20,25,30],"N2":[0.8,0.85,0.9,0.95],"N3":[0.975],"N4":[1.025]}
#print(list(ParameterGrid(param_grid)))
param_list=list(ParameterGrid(param_grid))
for x in range(len(param_list)):
    param=param_list[x]
    print(x)
    logger = logging.getLogger(str(x))  # logging对象
    fh = logging.FileHandler(str(x)+"a_test.log",mode="w")  # 文件对象
    sh = logging.StreamHandler()  # 输出流对象
    fm = logging.Formatter('%(asctime)s-%(filename)s[line%(lineno)d]-%(levelname)s-%(message)s')  # 格式化对象
    fh.setFormatter(fm)  # 设置格式
    sh.setFormatter(fm)  # 设置格式
    logger.addHandler(fh)  # logger添加文件输出流
    logger.addHandler(sh)  # logger添加标准输出流（std out）
    logger.setLevel(logging.INFO)  # 设置从那个等级开始提示
    #logging.basicConfig(filename=os.path.join(os.getcwd(), str(x)+'log.txt'), level=logging.INFO, format=LOG_FORMAT,filemode="a")
    cash_list, btcnum_list = random_wave_h(N1=param["N1"], N2=param["N2"], N3=param["N2"]*param["N3"], N4=param["N2"]*param["N4"], cash_list=[10000], btcnum_list=[0],
                                           status=1, price_stop=0, buy_price=0,max_price=0)

    df = pd.DataFrame({"cash": cash_list, "btc": btcnum_list})
    df.to_csv(str(x)+"_"+"a_random.csv")





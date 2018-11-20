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
from lib.myfun import *
import copy
import math
import os
import pdb
import re
import itertools
import numpy as np
import pandas as pd
import datetime as dt
import seaborn as sns
from arch import arch_model
import scipy.stats as stats
from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


exchange = 'BIAN'
symbols = ['btcusdt']

dataf = read_data(exchange, symbols[0], '1h', "2017-01-01", "2018-10-01")
print(dataf.head())

def GarmanKlass_get_estimator(data,window=30):
    a = 0.5 * np.log(data['high'] / data['low']) ** 2
    b = (2 * np.log(2) - 1) * (np.log(data['close'] / data['open']) ** 2)
    vol_mid1=a-b
    vol_mid1=ts_sum(vol_mid1,window)/window
    return np.sqrt(vol_mid1)*100

price_data=dataf
price_data.index=dataf["date"]

GarmanKlassVol=GarmanKlass_get_estimator(price_data)

price_data["vol"]=GarmanKlassVol
print(len(price_data))

ROLLING_WINDOW = 120  # 用到的滚动周期
VOL_NAME = 'GarmanKlassVol'  #计算的波动率名字，几个波动率都可以试一下，YangZhangVol最常用

def predict_AR(array, p=1):
    """第一种方法： 在给定滚动周期下利用AR(P)模型预测

    输入:
        df:DataFrame, 波动率原始数据
        window: 整数滚动周期
        p: int, lag of AR model
    输出:
        vols_pred: 时间序列, 预测波动率
    """

    #fit = lambda x: AR(x).fit(maxlag=p, disp=0).predict(start=x.size, end=x.size)
    #vols_pred = df[VOL_NAME].rolling(window).apply(fit)
    vols_pred=AR(array).fit(maxlag=p, disp=0).predict(start=array.size, end=array.size,dynamic=True)
    return vols_pred

def predict_AR_1(array, p=1):
    """第一种方法： 在给定滚动周期下利用AR(P)模型预测

    输入:
        df:DataFrame, 波动率原始数据
        window: 整数滚动周期
        p: int, lag of AR model
    输出:
        vols_pred: 时间序列, 预测波动率
    """

    #fit = lambda x: AR(x).fit(maxlag=p, disp=0).predict(start=x.size, end=x.size)
    #vols_pred = df[VOL_NAME].rolling(window).apply(fit)
    vols_pred=AR(array).fit(maxlag=20, disp=0).predict(start=array.size, end=array.size+29,dynamic=True)
    return vols_pred


#'''
def digui(data_test):
    predict_value=predict_AR(data_test[-120:])
    data_test=np.append(data_test,predict_value)
    if len(data_test)==150:
        return data_test
    return digui(data_test)

#'''
def get_mse(vols_true, vols_pred):
    """计算MSE, root mean squared error.

    输入:
        vols_true: 时间序列, 基准波动率
        vols_pred: 时间序列, 预测波动率
    输出:
        error: float, MSE
    """
    error = np.sqrt(mean_squared_error(vols_true, vols_pred))
    return error

base_data=price_data["vol"].shift(1)
base_data.fillna(base_data.iloc[1],inplace=True)

@tail_call_optimized
def digui_2(n,mse_list):
    if len(mse_list)>2000:
        return mse_list
    data_test=price_data["vol"].values[n:n+120]
    data_pre=digui(data_test)[-30:]
    mean_pre = np.mean(data_pre)
    mean_true = np.mean(base_data.values[n + 120:n + 150])
    ll=[mean_pre-mean_true]
    return digui_2(n+1,mse_list+ll)



@tail_call_optimized
def digui_3(n,mse_list):
    if len(mse_list)>2000:
        return mse_list
    data_test=price_data["vol"].values[n:n+120]
    data_pre=predict_AR_1(data_test)
    mean_pre=np.mean(data_pre)
    mean_true=np.mean(base_data.values[n+120:n+150])
    ll=[mean_pre-mean_true]
    return digui_3(n+1,mse_list+ll)




data_digui2=digui_2(30,[])
data_digui3=digui_3(30,[])

df=pd.DataFrame({"digui2":data_digui2,"digui3":data_digui3},index=range(len(data_digui2)))
print(df)
df.plot()
plt.show()



















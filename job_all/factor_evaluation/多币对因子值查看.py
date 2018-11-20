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

symbols = ['btcusdt',"ethusdt","xrpusdt","bccusdt","eosusdt","xlmusdt","ltcusdt","etcusdt","adausdt","trxusdt"]

'''
for symbol in symbols:
    data=pd.read_csv('../factor_writedb/multiple_subject/' + symbol+ "_" + "Alpha.alpha059_gtja4h" + '.csv', index_col=0)
    
'''
#data=pd.read_csv('../factor_writedb/multiple_subject/' + symbols[0]+ "_" + "Alpha.alpha059_gtja4h" + '.csv', index_col=0)
#print(data.head(40))





"""
data_test=data.ix[1:21]
print(data_test)
data_test[["high","open","low","close"]]=preprocessing.minmax_scale(data_test[["high","open","low","close"]])

data_test=data_test.iloc[:,:-1]
print(data_test)
Alpha = Alphas(data_test)
print(Alpha.alpha059().values[-1])
"""

a=list(range(1,202))
alpha_test=[]
for x in a:
    if x<10:
        alpha_test.append("Alpha.alpha00"+str(x))
    elif 10<x<100:
        alpha_test.append("Alpha.alpha0"+str(x))
    else:
        alpha_test.append("Alpha.alpha" + str(x))

@tail_call_optimized
def rolling_std(df,n,result_list=[],nan_len=20):
    if n+nan_len==len(df):
        mid_list=[np.nan]*nan_len
        mid_list=mid_list+result_list
        return mid_list
    data_test=copy.deepcopy(df.ix[n:n+nan_len])
    data_test[["high", "open", "low", "close"]] = preprocessing.minmax_scale(data_test[["high", "open", "low", "close"]])
    Alpha = Alphas(data_test)
    result_list.append(eval(alpha)().values[-1])
    return rolling_std(df,n+1,result_list)


for alpha in alpha_test[192:]:
    for symbol in symbols:
        try:
            data = pd.read_csv('../factor_writedb/multiple_subject/' + symbol + "_" + alpha +"_gtja4h" + '.csv',index_col=0)
            data["param_std"]=rolling_std(data,0,result_list=[],nan_len=60)
            data.to_csv("std_param_"+alpha+symbol+".csv")
            print(data.tail())
        except FileNotFoundError:
            pass

















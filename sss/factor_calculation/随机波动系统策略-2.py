# coding=utf-8

import sys
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


param_grid={"N1":[10,15,20,25,30],"N2":[0.8,0.85,0.9,0.95],"N3":[0.975],"N4":[1.025]}
#print(list(ParameterGrid(param_grid)))
param_list=list(ParameterGrid(param_grid))
#print(len(param_list))
#print(param_list[19])
#data=pd.read_csv("19_new_random.csv",index_col=0)

#'''
win_list=[2,4,6,8,9,12,13,16,17]
for i in win_list:
    data = pd.read_csv(str(i)+"_new_a_random.csv", index_col=0)
    data["asset"]=data["cash"]+data["btc"]*data["close"]
    print(param_list[i])
    data[["asset","close"]].plot()
    plt.show()
#'''
print("---"*20)



#'''
loss_list=[0,1,3,5,7,10,11,14,15,18,19]
for i in loss_list:
    data = pd.read_csv(str(i)+"_new_a_random.csv", index_col=0)
    data["asset"] = data["cash"] + data["btc"] * data["close"]
    print(i)
    print(param_list[i])
    data[["asset", "close"]].plot()
    plt.show()
#'''

'''
exchange = 'BIAN'
symbols = ['btcusdt']

dataf_h = read_data(exchange, symbols[0], '1h', "2017-01-01", "2018-10-01")
dataf_h.index=dataf_h["date"]

data_test=pd.read_csv("0_a_random.csv",index_col=0)


param=param_list[4]
N1=param["N1"]
dataf_current=dataf_h.ix[N1:9730+N1]
data_test.index=dataf_current.index
data_test["close"]=dataf_current["close"]

data_test["asset"]=data_test["cash"]+data_test["btc"]*data_test["close"]
print(data_test.head())

data_test[["close","asset"]].plot()
plt.show()
'''

















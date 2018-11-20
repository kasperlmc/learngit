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

exchange = 'BIAN'
symbols = ['btcusdt']

dataf_h = read_data(exchange, symbols[0], '1h', "2017-01-01", "2018-10-01")
dataf_h.index=dataf_h["date"]

param_grid={"N1":[10,15,20,25,30],"N2":[0.8,0.85,0.9,0.95],"N3":[0.975],"N4":[1.025]}
#print(list(ParameterGrid(param_grid)))
param_list=list(ParameterGrid(param_grid))
print(param_list)
for i in range(len(param_list)):
    data=pd.read_csv(str(i)+"_a_random.csv",index_col=0)
    param=param_list[i]
    N1=param["N1"]
    data_start = dataf_h.ix[N1]
    dataf_current=dataf_h.ix[N1:9730+N1]
    data.index=dataf_current.index
    data["close"]=dataf_current["close"]
    data.to_csv(str(i)+"_new_a_random.csv")


























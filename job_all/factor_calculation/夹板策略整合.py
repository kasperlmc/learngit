# coding=utf-8

import sys
import os
sys.path.append('..')
from lib.myfun import *
import pandas as pd
import numpy as np
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt

#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)

exchange = 'BIAN'
symbols = ['btcusdt']
dataf_h = read_data(exchange, symbols[0], '1h', "2017-01-01", "2018-10-01")
dataf_h.index=dataf_h["date"]

param_grid={"x":[0.8,0.85,0.9,0.95],"y":[0.015,0.02,0.05,0.075]}
#print(list(ParameterGrid(param_grid)))
param_list=list(ParameterGrid(param_grid))
print(param_list)

for i in range(len(param_list)):
    data=pd.read_csv(str(i)+"_jiaban.csv",index_col=0)
    param=param_list[i]
    print(param)
    dataf_current = dataf_h.ix[0:9730]
    data.index = dataf_current.index
    data["close"] = dataf_current["close"]
    data["asset"]=data["cash"]+data["btc"]*data["close"]
    data[["close","asset"]].plot()
    plt.show()

















# coding=utf-8
# 从数据库中取数据，计算price_volume因子，存入factor_writedb文件夹

import sys
sys.path.append('..')
from lib.myfun import *
from lib.factors import *
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

data=pd.read_csv("dtph.csv",index_col=0)
print(data.tail())
print(data.columns)
data["asset"]=data["cash"]+data["btcnum"]*data["close"]
data["net"]=data["asset"]/data["close"]
data["close_ss"]=ss.fit_transform(data[["close"]])
data.index=data["date"]
data[["net","close_ss"]].plot()
plt.show()

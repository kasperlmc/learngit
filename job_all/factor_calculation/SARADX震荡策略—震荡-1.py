# coding=utf-8

import sys
sys.path.append('..')
from lib.myfun import *
import copy
import pandas as pd
import numpy as np
import talib as ta
import copy
from lib.mutilple_factor_test import *
import matplotlib.pyplot as plt

#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)

data=pd.read_csv("saradx_long_xrp.csv",index_col=0)
data["date_time"]=data["date"]


data["asset"]=data["cash"]+data["btc"]*data["close"]
data["close_diff"]=data["close"].diff()
data["diff"]=np.sign(data["close_diff"].values)

data["net"]=data["asset"]-data["fee"]
data_net=data[["net","date_time","pos","close"]]
data_net.dropna(inplace=True)
data_net["date_time"] = pd.to_datetime(data_net["date_time"])

result,cols=summary_net(data_net,0,"saradx_long")
stat_ls=[result]
df=pd.DataFrame(stat_ls,columns=cols)


fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(data.index,data["asset"])
ax1.plot(data.index,data["close"],c="r")
ax2=ax1.twinx()
ax2.plot(data.index,data["vol"],c="g")
plt.show()



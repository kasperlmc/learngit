# coding=utf-8

import sys
sys.path.append('..')
import pandas as pd
import numpy as np
import talib as ta
from datetime import datetime
from numba import jit
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)
# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)


def timestamp_to_datetime(timestamp):
    """将 13 位整数的毫秒时间戳转化成本地普通时间格式)
    :param sp: 13 位整数的毫秒时间戳1456402864242)
    :return: 返回格式 {}2016-02-25 20:21:04.242000
    """
    local_dt_time = datetime.fromtimestamp(timestamp)
    return local_dt_time


data = pd.read_csv("/Users/wuyong/alldata/original_data/celue_all_coindata_result.csv",index_col=0)

data["date_time"] = [timestamp_to_datetime(x) for x in data["date"].values]
data["asset"] = data["asset"]/data["asset"].values[0]
print(data.head())

#
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(data["date_time"].values,data["close_btc"])
ax1.plot(data["date_time"].values,data["close_xrp"])
ax1.plot(data["date_time"].values,data["close_eos"])
ax1.plot(data["date_time"].values,data["close_eth"])
ax1.plot(data["date_time"].values,data["asset"])
ax1.scatter(data["date_time"].values, data["open"].values,marker="^",color="r")
ax1.scatter(data["date_time"].values, data["close"].values,marker="v",color="k")
ax1.annotate(str(data["asset"].values[-1]),xy=(data["date_time"].values[-1],data["asset"].values[-1]),xytext=(data["date_time"].values[-1],data["asset"].values[-1]))
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.legend()
plt.show()
plt.savefig("/Users/wuyong/alldata/original_data/test_all.png")

































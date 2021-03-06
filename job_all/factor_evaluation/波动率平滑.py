# coding=utf-8

import sys
sys.path.append('..')
from lib.myfun import *
from lib.factors import *
import copy
import math
import numpy as np
import pandas as pd
import talib as ta
import matplotlib.pyplot as plt

exchange = 'BIAN'
symbols = ['btcusdt']

dataf = read_data(exchange, symbols[0], '1h', "2017-01-01", "2018-10-01")
print(dataf.head())

def GarmanKlass_get_estimator(price_data, window=30, trading_periods=252, clean=True):
    "Garman-Klass（1980）利用了交易时段最高价、最低价和收盘价三个价格数据进行估计，该估计量通过将估计量除以调整 因子来纠正存在的偏差，以便得到方差的无偏估计。"
    log_hl = (price_data['high'] / price_data['low']).apply(np.log)
    log_cc = (price_data['close'] / price_data['close'].shift(1)).apply(np.log)

    rs = 0.5 * log_hl ** 2 - (2 * math.log(2) - 1) * log_cc ** 2

    def f(v):
        return (trading_periods * v.mean()) ** 0.5

    result = rs.rolling(window=window, center=False).apply(func=f)

    if clean:
        return result.dropna()
    else:
        return result

price_data=copy.deepcopy(dataf)
price_data.index=price_data["date"]

price_data["vol"]=GarmanKlass_get_estimator(price_data)


price_data["vol_ema"]=ta.MA(price_data["vol"],timeperiod=100)
print(price_data.head(40))
price_data[["vol_ema"]].plot()
plt.show()
































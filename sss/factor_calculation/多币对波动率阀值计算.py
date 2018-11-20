# coding=utf-8

import sys

sys.path.append('..')
from lib.myfun import *
import copy
import pandas as pd
import numpy as np
import talib as ta
import copy
import matplotlib.pyplot as plt
from lib.mutilple_factor_test import *

#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)

exchange = 'BIAN'
symbols = ['eosusdt']

dataf = read_data(exchange, symbols[0], '1h', "2017-01-01", "2018-10-01")
print(dataf.head())

def ts_sum(df ,window=10):
    return df.rolling(window).sum()

def GarmanKlass_Vol_1(data,window=30):
    a = 0.5 * np.log(data['high'] / data['low']) ** 2
    b = (2 * np.log(2) - 1) * (np.log(data['close'] / data['open']) ** 2)
    vol_mid1=a-b
    vol_mid1=ts_sum(vol_mid1,window)/window
    return np.sqrt(vol_mid1)*100

dataf["vol"]=GarmanKlass_Vol_1(dataf)

dataf[["vol"]].plot()
plt.show()





















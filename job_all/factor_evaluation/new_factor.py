# coding=utf-8

import sys
sys.path.append('..')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from lib.dataapi import *
import os
import talib as ta
import logging


errcode, errmsg, result = get_exsymbol("BIAN")
btc_list = [x for x in result if x[-3:] == "btc"]
print(btc_list)
print(len(btc_list))
for i in range(len(btc_list)):
    try:
        print(btc_list[i])
        symbol = btc_list[i]
        data_test = pd.read_csv('/Users/wuyong/alldata/factor_writedb/factor_stra_4h/BIAN_' + symbol + "_" + "alpha_202" + '.csv',index_col=0)
        print(data_test.head())
        factor = "Alpha.alpha202"
        data_test[factor] = data_test["consistence"]/data_test["volume"]
        factor_name = factor + "_" + "gtja4h"
        data_test.to_csv('/Users/wuyong/alldata/factor_writedb/factor_stra_4h/BIAN_' + symbol + "_" + factor_name + '.csv')
    except FileNotFoundError:
        pass

exit()
btc_list = ["bttbtc"]
for i in range(len(btc_list)):
    print(btc_list[i])
    symbol = btc_list[i]
    data_4h = pd.read_csv("/Users/wuyong/alldata/original_data/BIAN_"+symbol+"_4h_2018-01-01_2019-01-06.csv", index_col=0)
    data_5m = pd.read_csv("/Users/wuyong/alldata/original_data/BIAN_" + symbol + "_5m_2018-01-01_2019-01-08.csv", index_col=0)
    tickid_list = data_4h["tickid"].values
    result_list = []
    for tickid in tickid_list:
        data_5m_temp = data_5m[(data_5m["tickid"] >= tickid) & (data_5m["tickid"] < tickid+14400)]
        data_5m_temp["filter"] = [1 if abs(x-y) <= 0.95*abs(z-v) else 0 for x, y, z, v in zip(data_5m_temp["close"].values, data_5m_temp["open"].values, data_5m_temp["high"].values, data_5m_temp["low"].values)]
        data_5m_temp = data_5m_temp[data_5m_temp["filter"] == 1]
        volume_sum = data_5m_temp["volume"].values.sum()
        result_list.append(volume_sum)
    data_4h["consistence"] = result_list
    print(data_4h.head())
    print(data_4h.tail())
    data_4h.to_csv('/Users/wuyong/alldata/factor_writedb/factor_stra_4h/BIAN_' + symbol + "_" + "alpha_202" + '.csv')

























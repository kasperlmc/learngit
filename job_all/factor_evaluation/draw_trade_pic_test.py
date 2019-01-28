# coding=utf-8

import sys
sys.path.append('..')
import pandas as pd
import numpy as np
import talib as ta
from lib.draw_trade_pic import *
import warnings
warnings.filterwarnings(action="ignore")

pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)

for symble in ["btc", "xrp", "eth", "eos"]:
    data_trade = pd.read_csv("/Users/wuyong/alldata/original_data/"+symble+"usdt_traderesult.csv", index_col=0)
    data_k = pd.read_csv("/Users/wuyong/alldata/original_data/trades_bian_"+symble+"usdt_m_3mon.csv", index_col=0)
    data_k["tickid"] = data_k["tickid"]-60
    data_k["ma7"] = ta.MA(data_k["close"].values, 7)
    data_k["ma30"] = ta.MA(data_k["close"].values, 30)
    data_k["ma90"] = ta.MA(data_k["close"].values, 90)
    data_trade = data_trade[data_trade["position"] == 1]
    buy_time_list = data_trade["tickid"].values
    buy_price_list = data_trade["buy_price"].values
    print(data_trade.head())
    print(buy_price_list)
    print(buy_time_list)
    print(data_k.head())
    print(data_k.tail())
    tuple_list = [(x-x % 60, y) for x, y in zip(buy_time_list, buy_price_list)]
    print(tuple_list)
    exit()

    save_trade_fig(tuple_list, data_k, "wave_90F", "trade_"+symble+"_wave_90F")


















































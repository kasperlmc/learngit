# coding=utf-8
# 从数据库中取数据，计算price_volume因子，存入factor_writedb文件夹

import sys
sys.path.append('..')
from lib.myfun import *
from lib.factors import *
import copy
import pandas as pd
import numpy as np

exchange = 'BIAN'
symbols = ['btcusdt']

dataf = read_data(exchange, symbols[0], '1m', "2017-01-01", "2018-10-01")
print(dataf.head(30))

cash_list=[]
btcnum_list=[]
signal_list=[]
fee_list=[]
fee = 0.00075
slippage = 0.00025
threshold=0.005
for rid,row in dataf.iterrows():
    i=rid
    if rid==0:
        cash_list.append(row.close)
        btcnum_list.append(1)
        signal_list.append(np.nan)
        fee_list.append(0)
    else:
        print(i)
        diffasset=(cash_list[-1]-btcnum_list[-1]*row.open)/2
        ratio=diffasset/cash_list[-1]
        if ratio>threshold:
            buyprice=row.open*(1+slippage)
            buyamount=diffasset/buyprice
            signal_list.append("b")
            fee_list.append(buyprice*buyamount*fee)
            btcnum_list.append(btcnum_list[-1]+buyamount)
            cash_list.append(cash_list[-1]-buyprice*buyamount)
        elif ratio<-threshold:
            sellprice=row.open*(1-slippage)
            sellamount=np.abs(diffasset)/sellprice
            signal_list.append("s")
            fee_list.append(sellprice*sellamount*fee)
            btcnum_list.append(btcnum_list[-1]-sellamount)
            cash_list.append(cash_list[-1]+sellprice*sellamount)
        else:
            signal_list.append(np.nan)
            fee_list.append(0)
            btcnum_list.append(btcnum_list[-1])
            cash_list.append(cash_list[-1])

dataf['cash']=cash_list
dataf["btcnum"]=btcnum_list
dataf["fee"]=fee_list
dataf["signal"]=signal_list
print(dataf)
dataf.to_csv("dtph.csv")

# coding=utf-8
# 从数据库中取数据，计算price_volume因子，存入factor_writedb文件夹

import sys
sys.path.append('..')
from lib.myfun import *
from lib.factors_gtja import *
from lib.dataapi import *
import numpy as np
import pandas as pd
import copy


def price_volume(df, corr_window):
    # 计算因子：量价相关系数
    factor = talib.CORREL(df['close'], df['volume'], corr_window)
    return factor


def build_col_name(factor_name, param):
    col_name = str(factor_name)
    for k, v in param.items():
        col_name += '_' + str(v)
    return col_name





a=list(range(1,202))
alpha_test=[]
for x in a:
    if x<10:
        alpha_test.append("Alpha.alpha00"+str(x))
    elif 10<x<100:
        alpha_test.append("Alpha.alpha0"+str(x))
    else:
        alpha_test.append("Alpha.alpha" + str(x))
#print(alpha_test)
#print(alpha_test)
#print(alpha_use)

# alpha_test = ["Alpha.alpha040"]
if __name__ == '__main__':

    # exchange = 'BITFINEX'
    symbols = ["ethbtc", "xrpbtc", "mdabtc", "eosbtc", "xlmbtc", "tusdbtc", "ltcbtc",
               "stratbtc", "trxbtc", "adabtc", "iotabtc", "xmrbtc", "bnbbtc", "dashbtc",
               "xembtc", "etcbtc", "neobtc", "ontbtc", "zecbtc", "wavesbtc", "btgbtc",
               "vetbtc", "qtumbtc", "omgbtc", "zrxbtc", "gvtbtc", "bchabcbtc", "bchsvbtc"]

    # errcode, errmsg, result = get_exsymbol("BIAN")
    # symbols = [x for x in result if x[-3:] == "btc"]
    # print(len(symbols))

    # symbols = [x.upper() for x in symbols]

    # symbols = ["bchabcbtc", "bchsvbtc"]

    # 因子是price_volume, 有一个参数corr_window, 参数取值10/25/50
    # 能形成三组因子时间序列：price_volume_10/25/50, 存入本地和数据库
    # factor_name = 'price_volume'
    # param_grid = {'corr_window': [10, 25, 50]}
    # param_lst = list(ParameterGrid(param_grid))
    for symbol in symbols:
        for factor in alpha_test:
            try:
                dataf = pd.read_csv("/Users/wuyong/alldata/original_data/BIAN_"+symbol+"_4h_2018-01-01_2019-01-09.csv",index_col=0)
                print(symbol)
                print(dataf.head())
                # exit()
                Alpha = Alphas(dataf)
                col_name = factor
                df_m = copy.deepcopy(dataf)
                df_m[col_name] = eval(factor)()
                factor_name = factor + "_" + "gtja4h"
                fname = '/Users/wuyong/alldata/factor_writedb/factor_stra_4h/BIAN_' + symbol + "_" + factor_name + '.csv'
                write_db(df_m, fname, False)
                print('write' + fname + '...')
            except AttributeError:
                print(factor)




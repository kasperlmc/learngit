# coding=utf-8
# 从数据库中取数据，计算price_volume因子，存入factor_writedb文件夹

import sys
sys.path.append('..')
from lib.myfun import *
from lib.factors_gtja import *
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


if __name__ == '__main__':

    exchange = 'BITFINEX'
    symbols = ['btcusdt',"ethusdt","xrpusdt","zecusdt","eosusdt","neousdt","ltcusdt","etcusdt","etpusdt","iotusdt"]

    # 因子是price_volume, 有一个参数corr_window, 参数取值10/25/50
    # 能形成三组因子时间序列：price_volume_10/25/50, 存入本地和数据库
    #factor_name = 'price_volume'
    #param_grid = {'corr_window': [10, 25, 50]}
    #param_lst = list(ParameterGrid(param_grid))
    for symbol in symbols:
        for factor in alpha_test:
            #col_name = build_col_name(factor_name, param)
            try:
                dataf = read_data(exchange, symbol, '1h', "2017-01-01", "2018-10-01")
                Alpha = Alphas(dataf)
                col_name=factor
                df_m=copy.deepcopy(dataf)
                df_m[col_name] = eval(factor)()
                factor_name=factor + "_" + "gtja1h"
                fname = '../factor_writedb/multiple_subject/' + symbol + '_' + factor_name + '.csv'
                write_db(df_m, fname, False)
                print('write' + fname + '...')
            except AttributeError:
                print(factor)




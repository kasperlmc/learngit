# coding=utf-8
# 从数据库中取数据，计算price_volume因子，存入factor_writedb文件夹

import sys
sys.path.append('..')
from lib.myfun import *
from lib.factors import *
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





a=list(range(1,61))
alpha_test=[]
for x in a:
    if x<10:
        alpha_test.append("Alpha.alpha00"+str(x))
    else:
        alpha_test.append("Alpha.alpha0"+str(x))
#print(alpha_test)
#print(alpha_use)

rank_params=[10,5,15,20]


if __name__ == '__main__':

    exchange = 'BITMEX'
    symbols = ['xbtusd']

    # 因子是price_volume, 有一个参数corr_window, 参数取值10/25/50
    # 能形成三组因子时间序列：price_volume_10/25/50, 存入本地和数据库
    #factor_name = 'price_volume'
    #param_grid = {'corr_window': [10, 25, 50]}
    #param_lst = list(ParameterGrid(param_grid))
    for symbol in symbols:
        for factor in alpha_test:
            #col_name = build_col_name(factor_name, param)
            try:
                dataf = read_data(exchange, symbol, '4h', "2017-01-01", "2018-10-01")
                Alpha = Alphas(dataf)
                col_name=factor
                df_m=copy.deepcopy(dataf)
                check_list=[]
                for i in rank_params:
                    check_list.append(eval(factor)(i))
                    if i!=10 and len(check_list[0].dropna())==len(check_list[1].dropna()):
                        pass
                    else:
                        df_m[col_name+ "_rank_" +str(i)] = eval(factor)(i)
                fname = '../factor_writedb/' + symbol +"4h"+ '_' + factor + '.csv'
                write_db(df_m, fname, False)
                print('write' + fname + '...')
            except AttributeError:
                print(factor)




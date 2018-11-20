import sys
sys.path.append('..')
from lib.myfun import *
from lib.factors_gtja import *
#from lib.factors import *
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

alpha_test=["Alpha.alpha118"]

if __name__ == '__main__':

    exchange = 'BIAN'
    symbols = ["btcusdt","ethusdt","eosbtc","htbtc","xrpbtc","ethbtc"]
    period=["15m","30m","1h","4h","1d"]

    # 因子是price_volume, 有一个参数corr_window, 参数取值10/25/50
    # 能形成三组因子时间序列：price_volume_10/25/50, 存入本地和数据库
    #factor_name = 'price_volume'
    #param_grid = {'corr_window': [10, 25, 50]}
    #param_lst = list(ParameterGrid(param_grid))
    for symbol in symbols:
        for factor in alpha_test:
            for t in period:
            #col_name = build_col_name(factor_name, param)
                dataf = read_data(exchange, symbol, t, "2017-01-01", "2018-10-22")
                Alpha = Alphas(dataf)
                col_name=factor
                df_m=copy.deepcopy(dataf)
                df_m[col_name] = eval(factor)()
                factor_name=factor + "_" + t
                fname = '../factor_writedb/' + symbol + '_' + factor_name + '.csv'
                write_db(df_m, fname, False)
                print('write' + fname + '...')

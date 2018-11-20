# coding=utf-8
import sys
import os
sys.path.append(os.getcwd())

try:
    from .dataapi import get_exsymbol_kline
except:
    from dataapi import get_exsymbol_kline

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
import scipy.stats as st
import talib
import time
from sqlalchemy import create_engine

pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 300)
pd.set_option('display.width', 500)
pd.set_option('display.max_colwidth', 100)


def mkfpath(folder, fname):
    try:
        os.mkdir(folder)
    except:
        pass
    fpath = folder + '\\' + fname
    return fpath


def read_data(exchange, symbol, period, start_day, end_day):
    # 从数据库或文件中读数据
    fname = f'{exchange}_{symbol}_{period}_{start_day}_{end_day}'
    fname = fname.replace('/', '_')
    fpath = mkfpath('.\\api_data', fname + '.csv')
    print(fpath)
    print('reading and writing %s...' % fname)

    # 如果没有文件，从数据库中读
    if not os.path.exists(fpath):
        ohlc = get_exsymbol_kline(exchange, symbol, period, start_day, end_day)[2]
        # 写入数据
        if not ohlc.empty:
            ohlc.to_csv(fpath)
    # 如果有文件，从文件中读
    else:
        ohlc = pd.read_csv(fpath, index_col=[0])

    ohlc['date'] = pd.to_datetime(ohlc['date'])

    # 观察数据缺失情况
    count = ohlc['date'].diff().value_counts()
    if len(count) > 1:
        print('Warning: discontinuous data')
        print(count.head(), '\n')

    return ohlc


def write_db(df, fname, to_db):
    # 写数据到文件或数据库
    # to_db: bool, if True:写数据到文件和数据库, if False: 写数据到文件
    df.to_csv(fname, float_format='%.8f')
    if to_db:
        enginestr = 'mysql+pymysql://dongsheng:rulaiShenzhang699!@149.28.94.32:3306/quantytest'
        engine = create_engine(enginestr)
        tname = 'factor'
        df.to_sql(tname, engine, index=False, if_exists='append')


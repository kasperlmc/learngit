# coding=utf-8
# 从数据库中取数据，计算price_volume因子，存入factor_writedb文件夹

import sys
sys.path.append('..')
from lib.myfun import *
from lib.factors import *
import copy
import math
import os
import pdb
import re
import itertools
import numpy as np
import pandas as pd
import datetime as dt
import seaborn as sns
from arch import arch_model
import scipy.stats as stats
from statsmodels.tsa.ar_model import AR
from scipy.stats import kurtosis
import matplotlib.pyplot as plt
from dateutil.parser import parse
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from collections import defaultdict
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, ElasticNet, ElasticNetCV, LinearRegression


exchange = 'BIAN'
symbols = ['bccusdt']

dataf = read_data(exchange, symbols[0], '1h', "2017-01-01", "2018-10-01")
print(dataf.head())


def Parkinson_get_estimator(price_data, window=30, trading_periods=252, clean=True):
    "Parkinson（1980）估计量采用了交易时段最高价和最低价两个价格数据，利用极差进行估计。该估计量使价格波动区间在一定假设下比基于收盘价的估计量更能有效地估计回报波动率。"

    rs = (1.0 / (4.0 * math.log(2.0))) * ((price_data['high'] / price_data['low']).apply(np.log)) ** 2.0

    def f(v):
        return trading_periods * v.mean() ** 0.5

    result = rs.rolling(window=window, center=False).apply(func=f)

    if clean:
        return result.dropna()
    else:
        return result


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


def RogersSatchell_get_estimator(price_data, window=30, trading_periods=252, clean=True):
    "Parkinson和Garman-Klass估计量之所以能提高估计效率，是因为它们依赖于一些不适用于真实市场的假设，尤其价格服从不带漂移项的几何布朗运动以及交易是连续的假设。Rogers –Satchell在一定程度上放宽了这些限制条件，引入带有漂移项的更优的估计量。"
    log_ho = (price_data['high'] / price_data['open']).apply(np.log)
    log_lo = (price_data['low'] / price_data['open']).apply(np.log)
    log_co = (price_data['close'] / price_data['open']).apply(np.log)

    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)

    def f(v):
        return trading_periods * v.mean() ** 0.5

    result = rs.rolling(
        window=window,
        center=False
    ).apply(func=f)

    if clean:
        return result.dropna()
    else:
        return result


def YangZhang_get_estimator(price_data, window=30, trading_periods=252, clean=True):
    "　Garman-Klass（1980）估计量无法解决价格序列中存在跳空开盘的情况。Yang-Zhang（2000）推导出了适用于价格跳空开盘的估计量，本质上是各种估计量的加权平均。 "
    log_ho = (price_data['high'] / price_data['open']).apply(np.log)
    log_lo = (price_data['low'] / price_data['open']).apply(np.log)
    log_co = (price_data['close'] / price_data['open']).apply(np.log)

    log_oc = (price_data['open'] / price_data['close'].shift(1)).apply(np.log)
    log_oc_sq = log_oc ** 2

    log_cc = (price_data['close'] / price_data['close'].shift(1)).apply(np.log)
    log_cc_sq = log_cc ** 2

    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)

    close_vol = log_cc_sq.rolling(window=window, center=False).sum() * (1.0 / (window - 1.0))
    open_vol = log_oc_sq.rolling(window=window, center=False).sum() * (1.0 / (window - 1.0))
    window_rs = rs.rolling(window=window, center=False).sum() * (1.0 / (window - 1.0))

    k = 0.34 / (1 + (window + 1) / (window - 1))
    result = (open_vol + k * close_vol + (1 - k) * window_rs).apply(np.sqrt) * math.sqrt(trading_periods)

    if clean:
        return result.dropna()
    else:
        return result

def Raw_get_estimator(price_data, window=30, trading_periods=252, clean=True):
    "最经常用的收盘价方差计算的波动率"
    log_return = (price_data['close'] / price_data['close'].shift(1)).apply(np.log)

    result = log_return.rolling(window=window, center=False).std() * math.sqrt(trading_periods)

    if clean:
        return result.dropna()
    else:
        return result


def Skew_get_estimator(price_data, window=30, clean=True):
    "偏度Skew代替波动率"
    log_return = (price_data['close'] / price_data['close'].shift(1)).apply(np.log)

    result = log_return.rolling(window=window, center=False).skew()

    if clean:
        return result.dropna()
    else:
        return result

def Kurtosis_get_estimator(price_data, window=30, clean=True):
    "峰度Kurtosis代替波动率"
    log_return = (price_data['close'] / price_data['close'].shift(1)).apply(np.log)

    result = log_return.rolling(window=window, center=False).kurt()

    if clean:
        return result.dropna()
    else:
        return result


def HodgesTompkins_get_estimator(price_data, window=30, trading_periods=252, clean=True):
    "参考文献：The Sampling Properties of Volatility Cones,  Stewart Hodges and Robert Tompkins, 《Journal of Derivatives》 , 2002：https://www.researchgate.net/publication/228431910_Volatility_Cones_and_Their_Sampling_Properties"

    log_return = (price_data['close'] / price_data['close'].shift(1)).apply(np.log)

    vol = log_return.rolling(
        window=window,
        center=False
    ).std() * math.sqrt(trading_periods)

    h = window
    n = (log_return.count() - h) + 1

    adj_factor = 1.0 / (1.0 - (h / n) + ((h ** 2 - 1) / (3 * n ** 2)))

    result = vol * adj_factor

    if clean:
        return result.dropna()
    else:
        return result


price_data=dataf
price_data.index=dataf["date"]

#计算、并把几个波动率综合在一起
ParkinsonVol=Parkinson_get_estimator(price_data)
GarmanKlassVol=GarmanKlass_get_estimator(price_data)
RogersSatchellVol=RogersSatchell_get_estimator(price_data)
YangZhangVol=YangZhang_get_estimator(price_data)
RawVol=Raw_get_estimator(price_data)
SkewVol=Skew_get_estimator(price_data)
KurtosisVol=Kurtosis_get_estimator(price_data)
HodgesTompkinsVol=HodgesTompkins_get_estimator(price_data)
#综合在一起对比下
zvol=pd.concat([ParkinsonVol,GarmanKlassVol,RogersSatchellVol,YangZhangVol,RawVol,SkewVol,KurtosisVol,HodgesTompkinsVol],axis = 1).dropna()
zvol.columns=['ParkinsonVol','GarmanKlassVol','RogersSatchellVol','YangZhangVol','RawVol','SkewVol','KurtosisVol','HodgesTompkinsVol']
#print(zvol.head(30))

df = pd.concat([zvol, price_data],axis=1)
df=df.dropna()
print(df.head(10))


# 定义需要的全局变量

ROLLING_WINDOW = 120  # 用到的滚动周期
VOL_NAME = 'GarmanKlassVol'  #计算的波动率名字，几个波动率都可以试一下，YangZhangVol最常用


def predict_BM(df):
    """基准：利用当前波动率预测

    输入:
        df: DataFrame, 波动率原始数据
    输出:
        vols_pred: 时间序列, 预测波动率
    """
    vols_pred = df[VOL_NAME].shift(1)
    vols_pred = vols_pred.fillna(vols_pred.iloc[1])
    vols_pred.name = 'benchmark'
    print("Benchmark prediction finished.")
    return vols_pred

vols_pred_BM=predict_BM(df)


def get_mse(vols_true, vols_pred):
    """计算MSE, root mean squared error.

    输入:
        vols_true: 时间序列, 基准波动率
        vols_pred: 时间序列, 预测波动率
    输出:
        error: float, MSE
    """
    vols_pred_drop = vols_pred.dropna()
    error = np.sqrt(mean_squared_error(vols_true[vols_pred_drop.index], vols_pred_drop))
    return error

def predict_AR(df, window=ROLLING_WINDOW, p=1):
    """第一种方法： 在给定滚动周期下利用AR(P)模型预测

    输入:
        df:DataFrame, 波动率原始数据
        window: 整数滚动周期
        p: int, lag of AR model
    输出:
        vols_pred: 时间序列, 预测波动率
    """

    fit = lambda x: AR(x).fit(maxlag=p, disp=0).predict(start=x.size, end=x.size)
    vols_pred = df[VOL_NAME].rolling(window).apply(fit)
    vols_pred.name = 'AR' + '_' + repr(window) + '_' + repr(p)
    print(vols_pred.name + " prediction finished.")
    return vols_pred


# 在支持python3.6版本的平台上是可以运行的，这里不能行！！！

def predict_GARCH(df, params_dict=None):
    """ Forecast next step volatility with expanding window using GARCH model
    """

    rolling_window = ROLLING_WINDOW
    if params_dict is None:
        params_dict = {'vol': 'GARCH', 'p': 1, 'o': 0, 'q': 1}
    fit1 = lambda x: arch_model(x, **params_dict).fit(disp='off', show_warning=False) \
        .forecast().residual_variance.ix[x.size - 1, 'h.1']
    vols_pred = df[VOL_NAME].rolling(rolling_window).apply(fit1)
    vols_pred = vols_pred.clip(0, vols_pred.quantile(0.98))
    vols_pred.name = 'GARCH'
    print("GARCH prediction finished.")
    return vols_pred



vols_pred_AR=predict_AR(df)
print(get_mse(vols_pred_BM,vols_pred_AR))
#print(vols_pred_AR)
df["vol_pre"]=vols_pred_AR
df[["vol_pre","GarmanKlassVol"]].plot()
plt.show()

'''
vols_pred_GARCH=predict_GARCH(df).dropna()
print(get_mse(vols_pred_BM,vols_pred_AR))
print(get_mse(vols_pred_BM,vols_pred_GARCH))
'''






































































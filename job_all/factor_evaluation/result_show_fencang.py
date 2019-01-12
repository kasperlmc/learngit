# coding=utf-8

import sys
sys.path.append('..')
import pandas as pd
import numpy as np
import talib as ta
import seaborn as sns
from datetime import datetime
from numba import jit
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)
# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)


def timestamp_to_datetime(timestamp):
    """将 13 位整数的毫秒时间戳转化成本地普通时间格式)
    :param sp: 13 位整数的毫秒时间戳1456402864242)
    :return: 返回格式 {}2016-02-25 20:21:04.242000
    """
    local_dt_time = datetime.fromtimestamp(timestamp)
    return local_dt_time


def period_return(df):
    df['net'] = df['mean_net']
    df = df.resample(rule="1d", on='date_time', label="right", closed="right").apply(
        {"date_time": "last", "net": "last"})
    fig, ax = plt.subplots(1, 5, figsize=(20, 4))
    scaler = np.array([0.006, 0.008, 0.01, 0.015, 0.03]) * 1
    for i, days in enumerate([30, 45, 90, 180]):
        print(days, 'days')
        df['ret'] = df['net'] / df['net'].shift(days) - 1
        df_tmp = df.dropna()
        sns.kdeplot(df_tmp['ret'], shade=True, ax=ax[i], bw=scaler[i])
        ax[i].legend().set_visible(False)
        ax[i].set_xlabel(str(days) + '-day return')
        ax[i].set_ylabel('')
        ax[i].set_yticklabels('')
        ax[i].tick_params(left=False)
        ax[i].axvline(-0.05, c='k', linestyle='--')
        ax[i].axvline(0, c='k', linestyle='--')
        ax[i].axvline(0.05, c='k', linestyle='--')
        ax[i].axvline(0.25, c='k', linestyle='--')
        print('ret < -0.05:', (df_tmp['ret'] < -0.05).sum() / len(df_tmp))
        print('-0.05 < ret < 0:', ((-0.05 < df_tmp['ret']) & (df_tmp['ret'] < 0)).sum() / len(df_tmp))
        print('0 < ret < 0.05:', ((0 < df_tmp['ret']) & (df_tmp['ret'] < 0.05)).sum() / len(df_tmp))
        print('0.05 < ret < 0.25:', ((0.05 < df_tmp['ret']) & (df_tmp['ret'] < 0.25)).sum() / len(df_tmp))
        print('ret > 0.25:', (0.25 < df_tmp['ret']).sum() / len(df_tmp))
    plt.show()
# for coinpair in ["btcusdt","eosusdt","ethusdt","xrpusdt"]:
#     data_temp = pd.read_csv("/Users/wuyong/alldata/original_data/"+coinpair+"_traderesult.csv", index_col=0)
#     print(data_temp.head())
#     print(len(data_temp))
#     data_temp.columns = ["tickid", "close_"+coinpair, "position_"+coinpair, "asset_"+coinpair]
#
#     if coinpair == "btcusdt":
#         data_all = data_temp
#     else:
#         data_all = data_all.merge(data_temp, on="tickid", how="left")
# print(data_all.head())
# data_all.to_csv("/Users/wuyong/alldata/original_data/all_traderesult.csv")


asset_list = ["asset_"+x for x in ["btcusdt","eosusdt","ethusdt","xrpusdt"]]
close_list = ["close_"+x for x in ["btcusdt","eosusdt","ethusdt","xrpusdt"]]
#
data_all = pd.read_csv("/Users/wuyong/alldata/original_data/all_traderesult.csv", index_col=0)
# print(data_all.head())
# print(data_all.tail())
data_all.fillna(method="ffill", inplace=True)
data_all["asset_all"] = data_all[asset_list].sum(axis=1)
# print(data_all.head())
data_all["date_time"] = [timestamp_to_datetime(x) for x in data_all["tickid"].values]

for x in close_list:
    data_all[x] = data_all[x]/data_all[x].values[0]

for coin in ["btcusdt", "eosusdt", "ethusdt", "xrpusdt"]:
    # data_all[data_all["position_"+coin] > 0]["position_"+coin] = data_all["close_"+coin]
    data_all["position_"+coin] = [y if x > 0 else np.nan for x, y in zip(data_all["position_"+coin].values, data_all["close_"+coin].values)]


print(data_all.head())
data_all["asset_all"] = data_all["asset_all"]/data_all["asset_all"].values[0]
data_all["mean_net"] = data_all["asset_all"]
data_all.index = data_all["date_time"]
df = data_all.resample(rule="1d", on='date_time',label="right", closed="right").apply(
                {"date_time": "last", "mean_net": "last"})
print(df.head(30))







period_return(data_all)

# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax1.plot(data_all["date_time"].values,data_all["close_btcusdt"])
# ax1.plot(data_all["date_time"].values,data_all["close_xrpusdt"])
# ax1.plot(data_all["date_time"].values,data_all["close_eosusdt"])
# ax1.plot(data_all["date_time"].values,data_all["close_ethusdt"])
# ax1.plot(data_all["date_time"].values,data_all["asset_all"])
# ax1.scatter(data_all["date_time"].values, data_all["position_btcusdt"].values,marker="v",color=["r","b"])
# ax1.scatter(data_all["date_time"].values, data_all["position_eosusdt"].values,marker="v",color=["r","b"])
# ax1.scatter(data_all["date_time"].values, data_all["position_ethusdt"].values,marker="v",color=["r","b"])
# ax1.scatter(data_all["date_time"].values, data_all["position_xrpusdt"].values,marker="v",color=["r","b"])
# plt.legend()
# plt.show()























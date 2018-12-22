# coding=utf-8

import sys
sys.path.append('..')
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from numba import jit
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)
# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)

min_max_scaler = preprocessing.MinMaxScaler()
ss = StandardScaler()
# 滑点设置
slippage = 0.000

coin_list = ["btcusdt", "xrpusdt", "eosusdt", "ethusdt"]
coin_list = ["close_"+x for x in coin_list]

data_zb = pd.read_csv("/Users/wuyong/alldata/original_data/celue_all_coindata.csv", index_col=0)
data_zb["btcusdt"] = data_zb["close_btcusdt"]/data_zb["close_btcusdt"].values[0]
data_zb["xrpusdt"] = data_zb["close_xrpusdt"]/data_zb[coin_list[1]].values[0]
data_zb["eosusdt"] = data_zb[coin_list[2]]/data_zb[coin_list[2]].values[0]
data_zb["ethusdt"] = data_zb[coin_list[3]]/data_zb[coin_list[3]].values[0]

print(data_zb.head(10))
print(data_zb.tail(10))


@jit()
def all_coin_celue(data_celue):
    cash_list = np.zeros(len(data_zb))
    asset_list = np.zeros(len(data_zb))
    btcnum_list = np.zeros(len(data_zb))
    date_list = np.zeros(len(data_zb))
    date_list[0] = data_celue[0][5]
    cash_list[0] = 10000.0
    asset_list[0] = 10000.0
    btcnum_list[0] = 0.0
    coin_kind = 0
    tradetimes = 0
    profittimes = 0
    profitnum = 0
    losstimes = 0
    lossnum = 0
    buy_price = 0
    open_list = np.array([np.nan]*len(data_zb))
    close_list = np.array([np.nan]*len(data_zb))
    for n in range(1, len(data_celue)):
        date_list[n] = data_celue[n][5]
        if n < 100:
            cash_list[n] = cash_list[n - 1]
            btcnum_list[n] = btcnum_list[n - 1]
            asset_list[n] = asset_list[n - 1]
        else:
            if btcnum_list[n - 1] == 0.0:
                position_btc = data_celue[n][2]
                position_xrp = data_celue[n][4]
                position_eos = data_celue[n][6]
                position_eth = data_celue[n][8]

                if position_btc == 1:
                    open_list[n] = data_celue[n][9]
                    print("开仓买入btcusdt")
                    print("买入时点为%s，买入时点价格为%s" % (data_celue[n][0], data_celue[n][1]))
                    buy_price = data_celue[n][1] * (1 + slippage)  # 按照当前逐笔数据的价格乘以一个滑点买入币对
                    btcnum = cash_list[n - 1] / buy_price  # 买入的数量根据现有的资金量计算出来
                    btcnum_list[n] = btcnum  # 币对持仓数目列表记录该次交易买入的数量
                    cash_list[n] = 0.0  # 币对现金账户列表记录当前账户所有的现金数量
                    asset_list[n] = cash_list[n] + btcnum_list[n] * data_celue[n][1]  # 资产列表记录所有的总资产=现金数目+币对数目*当前逐笔数据价格
                    coin_kind = 1
                elif position_xrp == 1:
                    open_list[n] = data_celue[n][10]
                    print("开仓买入xrpusdt")
                    print("买入时点为%s，买入时点价格为%s" % (data_celue[n][0], data_celue[n][3]))
                    buy_price = data_celue[n][3] * (1 + slippage)  # 按照当前逐笔数据的价格乘以一个滑点买入币对
                    btcnum = cash_list[n - 1] / buy_price  # 买入的数量根据现有的资金量计算出来
                    btcnum_list[n] = btcnum  # 币对持仓数目列表记录该次交易买入的数量
                    cash_list[n] = 0.0  # 币对现金账户列表记录当前账户所有的现金数量
                    asset_list[n] = cash_list[n] + btcnum_list[n] * data_celue[n][3]  # 资产列表记录所有的总资产=现金数目+币对数目*当前逐笔数据价格
                    coin_kind = 3
                elif position_eos == 1:
                    open_list[n] = data_celue[n][11]
                    print("开仓买入eosusdt")
                    print("买入时点为%s，买入时点价格为%s" % (data_celue[n][0], data_celue[n][5]))
                    buy_price = data_celue[n][5] * (1 + slippage)  # 按照当前逐笔数据的价格乘以一个滑点买入币对
                    btcnum = cash_list[n - 1] / buy_price  # 买入的数量根据现有的资金量计算出来
                    btcnum_list[n] = btcnum  # 币对持仓数目列表记录该次交易买入的数量
                    cash_list[n] = 0.0  # 币对现金账户列表记录当前账户所有的现金数量
                    asset_list[n] = cash_list[n] + btcnum_list[n] * data_celue[n][5]  # 资产列表记录所有的总资产=现金数目+币对数目*当前逐笔数据价格
                    coin_kind = 5
                elif position_eth == 1:
                    open_list[n] = data_celue[n][12]
                    print("开仓买入ethusdt")
                    print("买入时点为%s，买入时点价格为%s" % (data_celue[n][0], data_celue[n][7]))
                    buy_price = data_celue[n][7] * (1 + slippage)  # 按照当前逐笔数据的价格乘以一个滑点买入币对
                    btcnum = cash_list[n - 1] / buy_price  # 买入的数量根据现有的资金量计算出来
                    btcnum_list[n] = btcnum  # 币对持仓数目列表记录该次交易买入的数量
                    cash_list[n] = 0.0  # 币对现金账户列表记录当前账户所有的现金数量
                    asset_list[n] = cash_list[n] + btcnum_list[n] * data_celue[n][7]  # 资产列表记录所有的总资产=现金数目+币对数目*当前逐笔数据价格
                    coin_kind = 7
                else:
                    cash_list[n] = cash_list[n-1]
                    btcnum_list[n] = btcnum_list[n-1]
                    asset_list[n] = asset_list[n-1]

            else:
                if data_celue[n][coin_kind+1] == 0:
                    print("平仓卖出")
                    print("卖出时点为%s，卖出时点价格为%s" % (data_celue[n][0], data_celue[n][coin_kind]))
                    sell_price = data_celue[n][coin_kind] * (1 - slippage)  # 按照当前逐笔数据的价格加上滑点和手续费卖出所有持仓
                    print("平仓价格:%s" % sell_price)
                    cash_get = sell_price * btcnum_list[n - 1]  # 卖出所有持仓获得的现金
                    cash_list[n] = cash_get  # 现金账户列表更新
                    btcnum_list[n] = 0.0  # 币对数目账户列表更新
                    asset_list[n] = cash_list[n] + btcnum_list[n] * data_celue[n][coin_kind]  # 资产账户列表更新
                    if coin_kind == 1:
                        close_list[n] = data_celue[n][9]
                    elif coin_kind == 3:
                        close_list[n] = data_celue[n][10]
                    elif coin_kind == 5:
                        close_list[n] = data_celue[n][11]
                    else:
                        close_list[n] = data_celue[n][12]

                    if sell_price > buy_price:
                        profittimes += 1
                        profitnum += btcnum_list[n-1]*(sell_price-buy_price)
                        tradetimes += 1
                    elif sell_price < buy_price:
                        losstimes += 1
                        lossnum += btcnum_list[n-1]*(sell_price-buy_price)
                        tradetimes += 1
                    print(tradetimes, profitnum, profittimes, lossnum, losstimes)
                    print(asset_list[n])
                else:
                    cash_list[n] = cash_list[n - 1]
                    btcnum_list[n] = btcnum_list[n - 1]
                    asset_list[n] = cash_list[n] + btcnum_list[n] * data_celue[n][coin_kind]  # 资产账户列表更新
    return cash_list, asset_list, btcnum_list, tradetimes, \
            profitnum, profittimes, lossnum, losstimes,open_list,close_list

#
cash_list, asset_list, btcnum_list, \
tradetimes, profitnum, profittimes, \
lossnum, losstimes, open_list, close_list = all_coin_celue(data_zb.values)

data_result = pd.DataFrame({"close_btc": data_zb["btcusdt"].values, "asset": asset_list,
                            "date": data_zb["tickid"].values, "close_xrp": data_zb["xrpusdt"].values,
                            "close_eth": data_zb["ethusdt"].values, "close_eos": data_zb["eosusdt"].values,
                            "open": open_list, "close": close_list})
print(data_result.tail(20))

data_result.to_csv("/Users/wuyong/alldata/original_data/celue_all_coindata_result.csv")





































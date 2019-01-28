# coding=utf-8

import sys
sys.path.append('..')
import pandas as pd
import numpy as np
import talib as ta
import time
from numba import jit
from lib.draw_trade_pic import save_trade_fig
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)
# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 滑点设置
slippage = 0.000


def ts_sum(df, window=10):
    return df.rolling(window).sum()


def ts_max(df, window=10):
    return df.rolling(window).max()


def ts_min(df ,window=10):
    return df.rolling(window).min()


def ts_lowday(df, window=10):
    return (window-1)-df.rolling(window).apply(np.argmin)


# 数据准备，数据说明
data_zb = pd.read_csv("/Users/wuyong/alldata/original_data/trades_bian_eosusdt_s_3mon.csv", index_col=0)
data_zb["tickid"] = data_zb["dealtime"]
data_zb["close_s"] = data_zb["price"]
data_k = pd.read_csv("/Users/wuyong/alldata/original_data/trades_bian_eosusdt_m_3mon.csv", index_col=0)
# atr = ta.ATR(data_k['high'].values, data_k['low'].values, data_k['close'].values, timeperiod=7)
# data_k["atr_7"] = atr
data_k["growth"] = data_k["close"]-data_k["open"]
data_k["growth"] = data_k["growth"].apply(lambda x: 1 if x > 0 else 0)
data_k["growth"] = ts_sum(data_k["growth"])
data_k["high_55"] = ts_max(data_k["high"], window=55)
data_k["low_55"] = ts_min(data_k["low"],window=55)
data_k["low_25"] = ts_min(data_k["low"],window=25)
data_k["high_25"] = ts_max(data_k["high"], window=25)
data_k["low_15"] = ts_min(data_k["low"],window=15)
data_k["low_days"] = ts_lowday(data_k["low"], window=15)
data_k["ma7"] = ta.MA(data_k["close"].values, timeperiod=7, matype=0)
data_k["ma30"] = ta.MA(data_k["close"].values, timeperiod=30, matype=0)
data_k["ma90"] = ta.MA(data_k["close"].values, timeperiod=90, matype=0)
data_k["ma450"] = ta.MA(data_k["close"].values, timeperiod=450, matype=0)
data_k["ma90_diff"] = data_k["ma90"].diff()
data_k["ma90_diff_abs"] = ts_sum(data_k["ma90"].diff().abs(), window=20)
data_k["ma90_div"] = data_k["ma90"]/(data_k["ma90"].shift(1))
data_k["ma90_diff20"] = (data_k["ma90"]-data_k["ma90"].shift(20)).abs()
data_k["ou_func"] = [1 if (x+3*y > 4*z) and (x+a > 2*y) else 0 for x, y, z, a, b in zip(data_k["ma7"].values, data_k["ma7"].shift(2).values, data_k["ma7"].shift(1).values, data_k["ma7"].shift(4).values, data_k["ma7"].shift(8).values)]
print(len(data_k["ou_func"]))

data_zb = data_zb[["tickid", "close_s"]]
data_k = data_k[["tickid", "open", "high", "close", "low", "growth", "high_55",
                 "ma7", "ma90", "low_55", "low_days", "ma30", "ma90_diff", "low_25",
                 "high_25", "low_15", "ou_func", "ma90_div", "date", "ma90_diff_abs", "ma90_diff20", "ma450"]]

print(data_k.tail(10))
# data_k[["ma90_diff20"]].plot()
# plt.show()
# print(data_zb.tail(10))
# # print(len(data_zb), len(data_k))
#
data_combine = data_zb.merge(data_k,how="left",on="tickid")
data_combine.fillna(method="ffill",inplace=True)
#
data_combine["tupo_down_55"] = [1 if x < y else 0 for x, y in zip(data_combine["close_s"].values, data_combine["low_55"].shift(1).values)]
data_combine["tupo_down_15"] = [1 if x < y else 0 for x, y in zip(data_combine["close_s"].values, data_combine["low_15"].shift(1).values)]
data_combine["tupo_high_25"] = [1 if x > y else 0 for x, y in zip(data_combine["close_s"].values, data_combine["high_25"].shift(1).values)]
data_combine["tupo_high_55"] = [1 if x > y else 0 for x, y in zip(data_combine["close_s"].values, data_combine["high_55"].shift(1).values)]
data_combine["buy"] = [1 if (True) and (True) else 0 for x, y in zip(data_combine["low_days"].shift(1).values, data_combine["ma90_div"].shift(1).values)]
data_combine["sell_1"] = [1 if x<y else 0 for x, y in zip(data_combine["close_s"].values, data_combine["low_25"].shift(1).values)]
data_combine["sell_2"] = [1 if (x < y) and (z < 0) else 0 for x, y, z in zip(data_combine["close_s"].values, data_combine["ma30"].shift(1).values, data_combine["ma90_diff"].shift(1).values)]
# print(data_combine.head(100))

data_celue = data_combine[["tupo_down_55", "tupo_down_15", "tupo_high_25", "sell_1", "sell_2", "tickid", "close_s", "tupo_high_55", "buy", "ma90_diff_abs", "ma90_diff20"]]
# print(data_celue.head(100))

# data_celue = data_celue[data_celue["tickid"] > 1543680000]


@jit()
def numba_celue(data_celue):
    cash_list = np.zeros(len(data_zb))
    asset_list = np.zeros(len(data_zb))
    btcnum_list = np.zeros(len(data_zb))
    position_list = np.zeros(len(data_zb))
    buy_list = np.zeros(len(data_zb))
    date_list = np.zeros(len(data_zb))
    cash_list[0] = 10000.0
    asset_list[0] = 10000.0
    btcnum_list[0] = 0.0
    tradetimes = 0
    profittimes = 0
    profitnum = 0
    losstimes = 0
    lossnum = 0
    buy_price = 0
    tupo_l_15 = 0
    tupo_l_55 = 0
    tupo_h_25 = 0
    for n in range(1, len(data_celue)):
        date_list[n] = data_celue[n][5]
        if n < 100:
            cash_list[n] = cash_list[n - 1]
            btcnum_list[n] = btcnum_list[n - 1]
            asset_list[n] = asset_list[n - 1]
        else:
            if btcnum_list[n-1] == 0.0:
                if data_celue[n][0] == 1:
                    tupo_l_55 = 1
                    tupo_l_15 = 0
                    tupo_h_25 = 0

                if (tupo_l_55 == 1) and (data_celue[n][2] == 1):
                    tupo_h_25 = 1
                    tupo_l_55 = 0

                if (tupo_h_25 == 1) and (data_celue[n][1] == 1):
                    tupo_l_15 = 1
                    tupo_h_25 = 0

                if (data_celue[n][8] == 1) and (tupo_l_15 == 1) and (data_celue[n][7] == 1) and (data_celue[n][9] <= 0.002) and (data_celue[n][10] <= 0.0005):
                    position_list[n] = 1
                    buy_list[n] = data_celue[n][6]
                    tupo_l_55 = 0
                    tupo_l_15 = 0
                    tupo_h_25 = 0
                    print("开仓买入")
                    print("买入时点为%s，买入时点价格为%s" % (data_celue[n][5], data_celue[n][6]))
                    buy_price = data_celue[n][6] * (1 + slippage)  # 按照当前逐笔数据的价格乘以一个滑点买入币对
                    btcnum = cash_list[n-1] / buy_price  # 买入的数量根据现有的资金量计算出来
                    print("买入价格为:%s" % buy_price)
                    btcnum_list[n] = btcnum  # 币对持仓数目列表记录该次交易买入的数量
                    cash_list[n] = 0.0  # 币对现金账户列表记录当前账户所有的现金数量
                    asset_list[n] = cash_list[n] + btcnum_list[n] * data_celue[n][6]  # 资产列表记录所有的总资产=现金数目+币对数目*当前逐笔数据价格

                elif (tupo_l_15 == 1) and (data_celue[n][7] == 1):
                    tupo_l_15 = 0
                    tupo_l_55 = 0
                    tupo_h_25 = 0
                    cash_list[n] = cash_list[n - 1]
                    btcnum_list[n] = btcnum_list[n - 1]
                    asset_list[n] = asset_list[n - 1]

                else:
                    cash_list[n] = cash_list[n - 1]
                    btcnum_list[n] = btcnum_list[n - 1]
                    asset_list[n] = asset_list[n - 1]

            else:
                if (data_celue[n][3] == 1) or (data_celue[n][4] == 1):
                    print("平仓卖出")
                    print("卖出时点为%s，卖出时点价格为%s" % (data_celue[n][5], data_celue[n][6]))
                    sell_price = data_celue[n][6] * (1 - slippage)  # 按照当前逐笔数据的价格加上滑点和手续费卖出所有持仓
                    print("平仓价格:%s" % sell_price)
                    cash_get = sell_price * btcnum_list[n - 1]  # 卖出所有持仓获得的现金
                    cash_list[n] = cash_get  # 现金账户列表更新
                    btcnum_list[n] = 0.0  # 币对数目账户列表更新
                    asset_list[n] = cash_list[n] + btcnum_list[n] * data_celue[n][6]  # 资产账户列表更新
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
                    asset_list[n] = cash_list[n] + btcnum_list[n] * data_celue[n][6]  # 资产账户列表更新
    return cash_list, asset_list, btcnum_list, tradetimes, profitnum, profittimes, lossnum, losstimes, position_list,buy_list, date_list


print("eosusdt—————————————10——————04—————————————02—————01—————————")
start = time.time()
# print(numba_celue(data_zb.values,data_k.values))
cash_list,asset_list,btcnum_list,tradetimes,profitnum,profittimes,lossnum,losstimes, position_list, buy_list, date_list = numba_celue(data_celue.values)
end = time.time()
print("Elapsed (with compilation) = %s" % (end - start))
df_result = pd.DataFrame({"tickid": date_list, "buy_price": buy_list, "position": position_list})
df_result = df_result[df_result["position"] == 1]
print(len(df_result))
print(df_result.head(20))
print(df_result.tail(20))
buy_time_list = df_result["tickid"].values
buy_price_list = df_result["buy_price"].values
tuple_list = [(x-x % 60, y) for x, y in zip(buy_time_list, buy_price_list)]
print(tuple_list)
print(len(tuple_list))
data_k["tickid"] = data_k["tickid"]-60
save_trade_fig(tuple_list, data_k, "eosusdt_xt_mazouping_ping", "eos_xingtai", 60)





















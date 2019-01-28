# coding=utf-8

import sys
sys.path.append('..')
import pandas as pd
import numpy as np
import talib as ta
from lib.draw_trade_pic import save_trade_fig


# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)


def ts_sum(df, window=10):
    return df.rolling(window).sum()


data_k = pd.read_csv("/Users/wuyong/alldata/original_data/btcusdt_day_k.csv", index_col=0)
print(data_k.head())
print(len(data_k))
data_k.fillna(method="ffill", inplace=True)
# print(data_k)
data_k["ma20"] = ta.EMA(data_k["close"].values, 20)
data_k["ma60"] = ta.EMA(data_k["close"].values, 60)
data_k["ma120"] = ta.EMA(data_k["close"].values, 120)
data_k["ma20_dif"] = data_k["ma20"].diff()
data_k["kongtou_pailie"] = [1 if (x < y) and (y < z) and (h <= x or True) else 0 for x, y, z, h in zip(data_k["ma20"].values, data_k["ma60"].values, data_k["ma120"].values, data_k["high"].values)]
data_k["kongtou_sum"] = ts_sum(data_k["kongtou_pailie"], window=15)
print(data_k.tail())

data_celue = data_k[["high", "ma20_dif", "kongtou_sum", "tickid"]]


def xingtai_celue(data_celue):
    buyprice_list = np.zeros(len(data_celue))
    position_list = np.zeros(len(data_celue))
    date_list = np.zeros(len(data_celue))
    date_list[0] = data_celue[0][3]
    kt_pailie = 0
    buy_time = 0
    for n in range(1, len(data_celue)):
        date_list[n] = data_celue[n][3]
        if data_celue[n][2] > 5 and n-buy_time > 15 and data_celue[n][1] < 0:
            kt_pailie = 1
            buy_time = n

        if kt_pailie == 1 and data_celue[n][1] > 0:
            kt_pailie = 0
            position_list[n] = 1
            buyprice_list[n] = data_celue[n][0]
    return buyprice_list, position_list, date_list


buyprice_list, position_list, date_list = xingtai_celue(data_celue.values)
df_result = pd.DataFrame({"tickid": date_list, "buy_price": buyprice_list, "position": position_list})


# df_result = df_result[df_result["position"] == 1]
# buy_time_list = df_result["tickid"].values
# buy_price_list = df_result["buy_price"].values
# tuple_list = [(x, y) for x, y in zip(buy_time_list, buy_price_list)]
# print(tuple_list)
# print(len(tuple_list))
# print(data_k)
# save_trade_fig(tuple_list, data_k, "btcmarket_xt", "btc_xingtai", 86400)


df_result["less_ma"] = [0 if x < min(y, z, v) else 1 for x, y, z, v in zip(data_k["high"], data_k["ma20"], data_k["ma60"], data_k["ma120"])]
print(df_result)

less_ma_list = df_result["less_ma"].values

for x in range(1,len(df_result)):
    if position_list[x-1] == 1:
        if less_ma_list[x] == 1:
            position_list[x] = 1
        else:
            pass

df_result["position"] = position_list
del df_result["buy_price"]
del df_result["less_ma"]

print(df_result)
# df_result.to_csv("/Users/wuyong/alldata/original_data/day_position.csv")















































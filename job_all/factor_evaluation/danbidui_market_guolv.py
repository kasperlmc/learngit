# coding=utf-8

import sys


class TailRecurseException(Exception):
    def __init__(self, args, kwargs):
        self.args = args
        self.kwargs = kwargs


def tail_call_optimized(g):
    """
    This function decorates a function with tail call
    optimization. It does this by throwing an exception
    if it is it's own grandparent, and catching such
    exceptions to fake the tail call optimization.

    This function fails if the decorated
    function recurses in a non-tail context.
    """

    def func(*args, **kwargs):
        f = sys._getframe()
        # 为什么是grandparent, 函数默认的第一层递归是父调用,
        # 对于尾递归, 不希望产生新的函数调用(即:祖父调用),
        # 所以这里抛出异常, 拿到参数, 退出被修饰函数的递归调用栈!(后面有动图分析)
        if f.f_back and f.f_back.f_back and f.f_back.f_back.f_code == f.f_code:
            # 抛出异常
            raise TailRecurseException(args, kwargs)
        else:
            while 1:
                try:
                    return g(*args, **kwargs)
                except TailRecurseException as e:
                    args = e.args
                    kwargs = e.kwargs
    func.__doc__ = g.__doc__
    return func


sys.path.append('..')
import pandas as pd
import numpy as np
import talib as ta
import copy
from lib.factors_gtja import *

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)

slippage = 0.002

symbols = ["ethbtc", "xrpbtc", "mdabtc", "eosbtc", "xlmbtc", "tusdbtc", "ltcbtc",
           "stratbtc", "trxbtc", "adabtc", "iotabtc", "xmrbtc", "bnbbtc", "dashbtc",
           "xembtc", "etcbtc", "neobtc", "ontbtc", "zecbtc", "wavesbtc", "btgbtc",
           "vetbtc", "qtumbtc", "omgbtc", "zrxbtc", "gvtbtc", "bchabcbtc", "bchsvbtc"]


# symbols = sorted(symbols)

aim_symbols = ["ethbtc", "eosbtc", "xrpbtc", "trxbtc", "tusdbtc", "bchabcbtc", "bchsvbtc", "ontbtc", "ltcbtc", "adabtc", "bnbbtc"]

symbols_close = [x+"_close" for x in ["ethbtc", "eosbtc", "xrpbtc", "trxbtc", "tusdbtc", "bchabcbtc", "bchsvbtc", "ontbtc", "ltcbtc", "adabtc", "bnbbtc"]]


def ts_sum(df, window=10):
    return df.rolling(window).sum()


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


def pick_coin(alpha_list):
    df_symbols_last = None
    df = None
    for i in range(len(alpha_list)):
        # 计算出每个alpha的策略指标
        try:
            for symbol in symbols:
                data = pd.read_csv(
                    '/Users/wuyong/alldata/factor_writedb/factor_stra_4h/BIAN_' + symbol + "_" + alpha_list[i] + "_gtja4h" + '.csv',
                    index_col=0)
                data = data[data["tickid"] > 1530093600]
                data.drop_duplicates(subset="tickid", keep="last", inplace=True)
                data["ma20"] = ta.EMA(data["close"].values, 20)
                data["ma60"] = ta.EMA(data["close"].values, 60)
                data["ma120"] = ta.EMA(data["close"].values, 120)
                data["ma20_dif"] = data["ma20"].diff()
                data["kongtou_pailie"] = [1 if (x < y) and (y < z) and (h <= x or True) else 0 for x, y, z, h in
                                          zip(data["ma20"].values, data["ma60"].values, data["ma120"].values,
                                                data["high"].values)]
                data["kongtou_sum"] = ts_sum(data["kongtou_pailie"], window=15)
                data_celue = data[["high", "ma20_dif", "kongtou_sum", "tickid"]]
                buyprice_list, position_list, date_list = xingtai_celue(data_celue.values)
                less_ma_list = [0 if x < min(y, z, v) else 1 for x, y, z, v in zip(data["high"], data["ma20"], data["ma60"], data["ma120"])]
                for x in range(1, len(position_list)):
                    if position_list[x - 1] == 1:
                        if less_ma_list[x] == 1:
                            position_list[x] = 1
                        else:
                            pass
                data["position"] = position_list

                if symbol == symbols[0]:
                    df = pd.DataFrame({symbol: data[alpha_list[i]].values, symbol + "_close": data["close"].values,
                                       symbol + "_open": data["open"].values, "tickid": data["tickid"].values, symbol + "_position": data["position"].values},
                                      index=data["date"].values)
                else:
                    df_1 = pd.DataFrame({symbol: data[alpha_list[i]].values, symbol + "_close": data["close"].values,
                                         symbol + "_open": data["open"].values, symbol + "_position": data["position"].values}, index=data["date"].values)
                    df = df.merge(df_1, how="left", left_index=True, right_index=True)
            df_symbols = df[symbols]
            df_symbols = df_symbols.rank(axis=1,numeric_only=True,na_option="keep")
            if i == 0:
                df_symbols_last = df_symbols
            else:
                df_symbols_temp = df_symbols
                df_symbols_last = df_symbols_last+df_symbols_temp

        except (FileNotFoundError, TypeError) as e:
            print(e)
    df[symbols] = df_symbols_last
    return df


@tail_call_optimized
def multiple_factor(df,cash_list=[10000],asset_list=[10000],buy_list=[[]],coinnum_list=[{}],n=1,close_list=[[]],max_value_list=[0],posittion=1, win_times=0, price_list=[]):
    if n == len(df):
        return cash_list, asset_list, buy_list, coinnum_list, close_list, max_value_list, win_times

    df_last_all = df.ix[n - 1]
    df_last_min = df[symbols].ix[n - 1]
    max_symbols_list = [df_last_min.idxmax(), df_last_min.drop(labels=df_last_min.idxmax()).idxmax()]
    hold_symbols_list = buy_list[-1]
    aim_symbols_list = list(set(max_symbols_list).intersection(set(aim_symbols)))
    strong_symbols_list = []
    for y in range(len(aim_symbols_list)):
        if df_last_all[aim_symbols_list[y]+"_close"] > df_last_all[aim_symbols_list[y]+"_close25"] and df_last_all[aim_symbols_list[y]+"_position"] == 1:
            strong_symbols_list.append(aim_symbols_list[y])
    buy_symbols_list = list(set(strong_symbols_list).difference(set(hold_symbols_list)))
    sell_symbols_list = list(set(hold_symbols_list).difference(set(aim_symbols_list)))
    df_now_all = df.ix[n]
    cash_now = cash_list[-1]
    sell_symbols_list = sorted(sell_symbols_list)
    for i in range(len(sell_symbols_list)):
        sell_price = df_now_all[sell_symbols_list[i] + "_open"] * (1 - slippage)
        sell_amount = coinnum_list[-1][sell_symbols_list[i]]
        cash_get = sell_price*sell_amount
        cash_now += cash_get

    hold_symbols_list = list(set(hold_symbols_list).difference(set(sell_symbols_list)))

    coinnum_dict_buy = {}
    buy_symbols_list = sorted(buy_symbols_list)
    for x in range(len(buy_symbols_list)):
        buy_price = df_now_all[buy_symbols_list[x] + "_open"] * (1 + slippage)
        buy_amount = (cash_now/(len(max_symbols_list)-len(hold_symbols_list)))/buy_price
        coinnum_dict_buy[buy_symbols_list[x]] = buy_amount
        cash_now -= cash_now/(len(max_symbols_list)-len(hold_symbols_list))

    coinnum_dict_old = copy.deepcopy(coinnum_list[-1])
    coinnum_dict_old.update(coinnum_dict_buy)

    coinnum_dict_now = {}
    hold_symbols_list = hold_symbols_list + buy_symbols_list
    for a in range(len(hold_symbols_list)):
        coinnum_dict_now[hold_symbols_list[a]] = coinnum_dict_old[hold_symbols_list[a]]

    buy_list.append(hold_symbols_list)
    coinnum_list.append(coinnum_dict_now)
    cash_list.append(cash_now)
    asset = cash_list[-1]
    for b in range(len(hold_symbols_list)):
        asset = asset + coinnum_dict_now[hold_symbols_list[b]]*df_now_all[hold_symbols_list[b] + "_close"]
    asset_list.append(asset)
    return multiple_factor(df,cash_list, asset_list, buy_list, coinnum_list, n+1, close_list, max_value_list, posittion, win_times, price_list)


alpha_two_combine = [("Alpha.alpha003", "Alpha.alpha014", "Alpha.alpha028", "Alpha.alpha050","Alpha.alpha051",
                      "Alpha.alpha069", "Alpha.alpha096", "Alpha.alpha128","Alpha.alpha167", "Alpha.alpha175")]
df = pick_coin(alpha_two_combine[0])
print(df.head())
print(df.tail())
for close in symbols_close:
    df[close + str(25)] = ta.MA(df[close].values, timeperiod=25, matype=0)

df["date_time"] = df.index.values

cash_list, asset_list, buy_list, coinnum_list, close_list, max_value_list, win_times = multiple_factor(df)
df_result = pd.DataFrame({"cash": cash_list, "asset": asset_list}, index=df["date_time"])
print(df_result)














































































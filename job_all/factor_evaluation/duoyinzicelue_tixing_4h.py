# coding=utf-8

import sys
sys.path.append('..')
import pandas as pd
import numpy as np
from lib.factors_gtja import *
from lib.realtime_kline_api_all import *
from lib.notifyapi import *
import time
import datetime
from collections import Counter


symbols = ["ethbtc", "xrpbtc", "mdabtc", "eosbtc", "xlmbtc", "tusdbtc", "ltcbtc",
           "stratbtc", "trxbtc", "adabtc", "iotabtc", "xmrbtc", "bnbbtc", "dashbtc",
           "xembtc", "etcbtc", "neobtc", "ontbtc", "zecbtc", "wavesbtc", "btgbtc",
           "vetbtc", "qtumbtc", "omgbtc", "zrxbtc", "gvtbtc", "bchabcbtc", "bchsvbtc"]

cols = ["exchange", "period", "symbol", "tickid", "open", "high", "low", "close", "volume", "amount"]

# symbols = ["tusdbtc"]
slippage = 0.002


# 获取本地各个币对的K线数据
# def get_data_local(exchange, period, symbol, start_time, n=21):
#     data = pd.read_csv("/Users/wuyong/alldata/original_data/"+exchange+"_"+symbol+"_"+period+"_2018-01-01_2019-01-06.csv", index_col=0)
#     data.index = data["date"].values
#     if start_time in data.index:
#         data = data.loc[start_time:]
#         data = data.head(n)
#         return data
#     else:
#         return pd.DataFrame()

alpha_test = ["Alpha.alpha003", "Alpha.alpha014", "Alpha.alpha050","Alpha.alpha051",
              "Alpha.alpha069", "Alpha.alpha128","Alpha.alpha167", "Alpha.alpha175"]


# 获取上一根K线各个币对因子的排名
def get_rank(now_time, m=30, alpha=alpha_test):
    start_time = (now_time-now_time % 14400)-m*14400
    time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))
    symbol_upper = [symbol.upper() for symbol in symbols]
    starttime = datetime.datetime.now()
    df_all = get_realtime_data('BIAN', '4h', symbol_upper, start_time=time_str, end_time=None)
    endtime = datetime.datetime.now()
    print((endtime - starttime).seconds)
    df_all = pd.DataFrame(df_all, columns=cols)
    df_all[["open", "close", "high", "low", "volume", "amount"]] = df_all[
        ["open", "close", "high", "low", "volume", "amount"]].astype("float")
    results = []
    for i in range(len(alpha)):
        print(alpha[i])
        result_dict_temp = {}
        for symbol in symbols:
            print(symbol)
            df_temp = df_all[df_all["symbol"] == symbol]
            df_temp = df_temp.head(m)
            if len(df_temp) == 0:
                pass
            else:
                df_temp.index = range(len(df_temp))
                Alpha = Alphas(df_temp)
                df_temp[alpha[i]] = eval(alpha[i])()
                result_dict_temp[symbol] = df_temp[alpha[i]].dropna().values[-1]
        results.append(result_dict_temp)
    result_dict_last = {}
    for _ in results:
        for k, v in _.items():
            result_dict_last.setdefault(k, []).append(v)
    df = pd.DataFrame.from_dict(result_dict_last)
    df = df.rank(axis=1, numeric_only=True, na_option="keep")
    series = df.sum()
    max_symble = series.idxmax()
    return max_symble


# 获取币对当前的价格
def get_now_price(now_time, coin):
    tepm_k_tickid = now_time
    time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(tepm_k_tickid))
    values_list = get_realtime_data('BIAN', '1m', coin.upper(), start_time=time_str, end_time=None)
    df_temp = pd.DataFrame(values_list, columns=cols)
    df_temp[["open", "close", "high", "low", "volume", "amount"]] = df_temp[["open", "close", "high", "low", "volume", "amount"]].astype("float")
    df_temp = df_temp[df_temp["tickid"] == tepm_k_tickid]
    return df_temp["open"].values[0]


# 获取币对当前25日均价
def get_now_ma25(now_time, coin, ma=25):
    start_time = (now_time - now_time % 14400) - ma * 14400
    time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))
    values_list = get_realtime_data('BIAN', '4h', coin.upper(), start_time=time_str, end_time=None)
    df_temp = pd.DataFrame(values_list, columns=cols)
    df_temp[["open", "close", "high", "low", "volume", "amount"]] = df_temp[
        ["open", "close", "high", "low", "volume", "amount"]].astype("float")
    df_temp = df_temp.head(25)
    ma = df_temp["close"].mean()
    return ma


# 获取上一次策略运行之后的结果
def get_last_result():
    data_result = pd.read_csv("/Users/wuyong/alldata/original_data/last_result.csv", index_col=0)
    cash = data_result.tail(1)["cash"].values[0]
    coin = data_result.tail(1)["coin"].values[0]
    coin_num = data_result.tail(1)["coin_num"].values[0]
    asset = data_result.tail(1)["asset"].values[0]
    date = data_result.tail(1)["date"].values[0]
    buyprice = data_result.tail(1)["buy_price"].values[0]
    return cash, coin, coin_num, asset, date, buyprice


# 依据各个币对的因子排名结果进行本次交易
def multi_factor(cash, coin, coin_num, now_time, date, buyprice):
    max_coin = get_rank(now_time)
    print("此时因子值最大的币对为：%s" % max_coin)
    if max_coin in ["ethbtc", "eosbtc", "xrpbtc", "trxbtc","tusdbtc", "bchabcbtc", "bchsvbtc", "ontbtc", "ltcbtc", "adabtc", "bnbbtc"]:
        coin_ma = get_now_ma25(now_time, max_coin)
        if coin == max_coin:
            asset = get_now_price(now_time, coin)*coin_num
            return cash, coin, coin_num, asset, date, buyprice
        else:
            if coin_num > 0:
                sell_price = get_now_price(now_time, coin)*(1-slippage)
                cash_get = sell_price*coin_num
                if get_now_price(now_time, max_coin) > coin_ma:
                    buy_price = get_now_price(now_time, max_coin)*(1+slippage)
                    coin_num = cash_get/buy_price
                    asset = cash_get
                    print("策略转仓")
                    print("之前持有币对为：%s，现在持仓币对为：%s" % (coin, max_coin))
                    date = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now_time))
                    buyprice = buy_price
                    return cash, max_coin, coin_num, asset, date, buyprice
                else:
                    coin = np.nan
                    coin_num = 0
                    asset = cash_get
                    print("该币对当前不宜开仓，策略平仓，此前持仓币对为：%s" % coin)
                    return cash_get, coin, coin_num, asset, date, buyprice
            else:
                if get_now_price(now_time, max_coin) > coin_ma:
                    buy_price = get_now_price(now_time, max_coin)*(1+slippage)
                    coin_num = cash/buy_price
                    asset = cash
                    cash = 0
                    print("策略开仓")
                    print("本次开仓买入币对为：%s" % max_coin)
                    date = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now_time))
                    buyprice = buy_price
                    return cash, max_coin, coin_num, asset, date, buyprice
                else:
                    print("该目标币对当前不宜开仓")
                    asset = cash
                    return cash, coin, coin_num, asset, date, buyprice
    else:
        if coin_num > 0:
            print("策略平仓")
            print("本次平仓卖出币对为：%s" % coin)
            sell_price = get_now_price(now_time, coin)*(1-slippage)
            cash_get = sell_price*coin_num
            coin = np.nan
            coin_num = 0
            asset = cash_get
            return cash_get, coin, coin_num, asset, date, buyprice

        else:
            asset = cash
            return cash, coin, coin_num, asset, date, buyprice


# 更新策略的运行结果
def renew_result(cash, coin, coin_num, asset, date, buyprice):
    data_result = pd.read_csv("/Users/wuyong/alldata/original_data/last_result.csv", index_col=0)
    data_temp = pd.DataFrame({"cash": cash, "coin": coin, "coin_num": coin_num, "asset": asset, "date": date, "buy_price": buyprice}, index=range(1))
    data_result = pd.concat([data_result, data_temp], axis=0, ignore_index=True)
    return data_result


# get_rank(1546992060)
# get_now_price(1546992060, "adabtc")
# get_last_result()
# time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(1546992060))
# print(time_str)
# get_now_ma25(1546992060, "adabtc")


# for i in range(1546963260, 1547006460, 14400):
#     print(i)
#     now_time = i
#     cash, coin, coin_num, asset = get_last_result()
#     print(cash)
#     cash, coin, coin_num, asset = multi_factor(cash, coin, coin_num, now_time)
#     data_new = renew_result(cash, coin, coin_num, asset)
#     data_new.to_csv("/Users/wuyong/alldata/original_data/last_result.csv")


df = pd.DataFrame({"cash": 10000, "coin": np.nan, "coin_num": 0, "asset": 10000, "date": np.nan, "buy_price": 0}, index=range(1))
df.to_csv("/Users/wuyong/alldata/original_data/last_result.csv")
# # df_1 = pd.DataFrame({"cash":10000, "coin":np.nan, "coin_num":0, "asset": 10000}, index=range(1))
# # print(pd.concat([df, df_1], axis=0, ignore_index=True))

# 获取当前时间
now_time = int(time.time())
time_str_tmp = time.strftime('%Y-%m-%d %H:%M', time.localtime(now_time))
now_time = int(time.mktime(time.strptime(time_str_tmp, '%Y-%m-%d %H:%M')))
now_time = 1550534460
time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now_time))
print(time_str)

# 获取上次程序运行的结果
cash, coin, coin_num, asset, date, buyprice = get_last_result()

# 结合上次运行结果，开始程序的该次运行
cash_new, coin_new, coin_num_new, asset_new, date_new, buyprice_new = multi_factor(cash, coin, coin_num, now_time, date, buyprice)

# 更新程序运行结果
data_new = renew_result(cash_new, coin_new, coin_num_new, asset_new, date_new, buyprice_new)
data_new.to_csv("/Users/wuyong/alldata/original_data/last_result.csv")

net_asset = asset_new/10000
print("当前该策略的净值为：%s" % net_asset)

if time_str[11:13] == "08" or True:
    if coin_num_new > 0:
        print("当前策略持仓币对为：%s, 该币对的买入时间为%s" % (coin_new, date_new))
        now_price = get_now_price(now_time, coin_new)
        margin = (now_price-buyprice_new)/buyprice_new
        print("该币对当前上涨幅度为：%s" % margin)
    else:
        print("当前策略为空仓状态")





















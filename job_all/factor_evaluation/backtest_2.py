import pandas as pd
import numpy as np
import sys
import datetime


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


# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)

data_all_btc = pd.read_csv("/Users/wuyong/alldata/original_data/bitfinex_btcusdt_1h_all.csv", index_col=0)
data_all_ltc = pd.read_csv("/Users/wuyong/alldata/original_data/bitfinex_ltcusdt_1h_all.csv", index_col=0)


train_end_date = "2018-01-01 00:00:00"
# train_end_date = datetime.datetime.strptime(train_end_date, "%Y-%m-%d %H:%M:%S")
# time_delta = datetime.timedelta(hours=8)
# train_end_date = train_end_date-time_delta
# print(train_end_date)
# train_end_date = "2017-12-31 16:00:00"
data_train_btc = data_all_btc[data_all_btc["date"] < train_end_date]
data_train_ltc = data_all_ltc[data_all_ltc["date"] < train_end_date]

data_test_btc = data_all_btc[data_all_btc["date"] >= train_end_date]
test_k_high_btc = data_test_btc["high"].values
test_k_open_btc = data_test_btc["open"].values
test_k_low_btc = data_test_btc["low"].values
test_k_close_btc = data_test_btc["close"].values

data_test_ltc = data_all_ltc[data_all_ltc["date"] >= train_end_date]

coin_list = ["btcusdt", "ltcusdt"]

open_df = pd.DataFrame()
close_df = pd.DataFrame()
high_df = pd.DataFrame()
low_df = pd.DataFrame()

symble_dict = {}
for coin in coin_list:
    data = pd.read_csv("/Users/wuyong/alldata/original_data/bitfinex_" + coin + "_1h_all.csv", index_col=0)
    data.fillna(method="ffill", inplace=True)
    data.fillna(0, inplace=True)
    data.index = data["date"].values
    open_df = data["open"]
    close_df = data["close"]
    high_df = data["high"]
    low_df = data["low"]
    symble_dict[coin] = [open_df.values, close_df.values, high_df.values, low_df.values]

length_k = 80
length_hold = 20
sllippage = 0.002

open_list_test = data_test_btc["open"].values
date_list_test = data_test_btc["date"].values
start_date = date_list_test[0]
aa = len(data_test_btc)-length_hold
print(aa)


@tail_call_optimized
def stra_func(cash_list=[10000], asset_list=[10000], buy_num_list=[0], date_list=[start_date], n=0, trade_price=[0], ret_list=[0], t_list=[0]):
    if length_k+length_hold*n > aa:
        return cash_list, asset_list, buy_num_list, date_list, trade_price, ret_list, t_list

    date_list.append(date_list_test[length_k+length_hold*n])

    test_k_high = test_k_high_btc[length_hold*n:length_k+length_hold*n]
    test_k_open = test_k_open_btc[length_hold*n:length_k+length_hold*n]
    test_k_low = test_k_low_btc[length_hold*n:length_k+length_hold*n]
    test_k_close = test_k_close_btc[length_hold*n:length_k+length_hold*n]

    # dt = pd.DataFrame(columns=["coin", "T", "ret"])
    # y = 0
    # num = len(open_df) - 1
    # starttime = datetime.datetime.now()
    # for d in range(length_k, num - length_hold, length_hold):
    #     close2 = close_df.iloc[d - length_k+1:d + 1]
    #     open2 = open_df.iloc[d - length_k+1:d + 1]
    #     high2 = high_df.iloc[d - length_k+1:d + 1]
    #     low2 = low_df.iloc[d - length_k+1:d + 1]
    #     for coin in coin_list:
    #         corropen = round(np.corrcoef(test_k_open, open2["open_" + coin])[0][1], 3)
    #         corrclose = round(np.corrcoef(test_k_close, close2["close_" + coin])[0][1], 3)
    #         corrhigh = round(np.corrcoef(test_k_high, high2["high_" + coin])[0][1], 3)
    #         corrlow = round(np.corrcoef(test_k_low, low2["low_" + coin])[0][1], 3)
    #         return_20 = close_df["close_" + coin].values[d + length_hold] / close_df["close_" + coin].values[d]
    #         T = (corrclose + corropen + corrhigh + corrlow) / 4
    #         dt.loc[y] = [coin, T, return_20]
    #         y += 1
    # dt.fillna(0, inplace=True)
    # dt.sort_values(by="T", ascending=False, inplace=True)
    # endtime = datetime.datetime.now()
    # print((endtime - starttime).seconds)

    dt = pd.DataFrame(columns=["coin", "T", "ret"])
    y = 0
    for coin in coin_list:
        close_values_all = symble_dict[coin][1]
        open_values_all = symble_dict[coin][0]
        high_values_all = symble_dict[coin][2]
        low_values_all = symble_dict[coin][3]
        num = len(open_values_all) - 1
        for d in range(length_k, num - length_hold, length_hold):
            close2 = close_values_all[d - length_k + 1:d + 1]
            open2 = open_values_all[d - length_k + 1:d + 1]
            high2 = high_values_all[d - length_k + 1:d + 1]
            low2 = low_values_all[d - length_k + 1:d + 1]
            corropen = round(np.corrcoef(test_k_open, open2)[0][1], 3)
            corrclose = round(np.corrcoef(test_k_close, close2)[0][1], 3)
            corrhigh = round(np.corrcoef(test_k_high, high2)[0][1], 3)
            corrlow = round(np.corrcoef(test_k_low, low2)[0][1], 3)
            return_20 = close_values_all[d + length_hold] / close_values_all[d]
            T = (corrclose + corropen + corrhigh + corrlow) / 4
            dt.loc[y] = [coin, T, return_20]
            y += 1
    dt.fillna(0, inplace=True)
    dt.sort_values(by="T", ascending=False, inplace=True)
    # filter_dt = dt[dt["T"] >= 0.8]
    t_list.append(dt.iloc[0]["T"])
    ret_list.append(dt.iloc[0]["ret"])

    if buy_num_list[-1] > 0:
        if dt.iloc[0]["ret"] > 1.002 and dt.iloc[0]["T"] > 0.8:
            print("继续持仓")
            cash_list.append(cash_list[-1])
            buy_num_list.append(buy_num_list[-1])
            asset_list.append(cash_list[-1]+buy_num_list[-1]*open_list_test[length_k+length_hold*n])
            trade_price.append(0)
        else:
            print("平仓")
            sell_price = open_list_test[length_k+length_hold*n]*(1-sllippage)
            sell_amount = buy_num_list[-1]
            cash_get = sell_price*sell_amount
            cash_list.append(cash_list[-1]+cash_get)
            buy_num_list.append(0)
            asset_list.append(cash_list[-1])
            trade_price.append(-1*open_list_test[length_k+length_hold*n])

    else:
        if dt.iloc[0]["ret"] > 1.002 and dt.iloc[0]["T"] > 0.8:
            print("开仓")
            buy_price = open_list_test[length_k+length_hold*n]*(1+sllippage)
            buy_amount = cash_list[-1]/buy_price
            cash_list.append(0)
            buy_num_list.append(buy_amount)
            asset_list.append(buy_num_list[-1]*test_k_close[-1])
            trade_price.append(open_list_test[length_k+length_hold*n])
        else:
            print("不动")
            buy_num_list.append(buy_num_list[-1])
            cash_list.append(cash_list[-1])
            asset_list.append(asset_list[-1])
            trade_price.append(0)
    return stra_func(cash_list, asset_list, buy_num_list, date_list, n+1, trade_price, ret_list, t_list)


cash_list, asset_list, buy_num_list, date_list, trade_price, ret_list, t_list = stra_func(cash_list=[10000], asset_list=[10000], buy_num_list=[0], date_list=[start_date], n=0)

df_result_day = pd.DataFrame({"asset": asset_list, "cash": cash_list, "price": trade_price, "coinnum": buy_num_list, "ret": ret_list, "t": t_list}, index=date_list)
print(df_result_day)
df_result_day.to_csv("/Users/wuyong/alldata/original_data/bitfinex_btcusdt_1h_all_result.csv")























































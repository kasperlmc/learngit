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
import matplotlib.pyplot as plt
import numpy as np
from lib.factors_gtja import *
import os
import talib as ta
from lib.dataapi import *
import copy
import time
import datetime
import mpl_finance as mpf


# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)

"""
下面是一系列的评价指标，其中主要用到的是总收益、夏普比例、胜率，交易次数等
"""


def total_ret(net):
    # print(net.iloc[-1],net.iloc[0])
    return net.iloc[-1] / net.iloc[0] - 1


def long_ret(net_df):
    net_df = net_df.copy()
    net_df['ret'] = net_df['net'] / net_df['net'].shift(1)
    long_ret = np.prod(net_df['ret']) - 1
    return long_ret


def month_profit(net_df):
    net_df = net_df.set_index('date_time')
    ret = []
    for gid, group in net_df.groupby(pd.Grouper(freq='M')):
        gid = group.index[0]
        long_return = long_ret(group)
        ret.append([gid, long_return])

    month_ret = pd.DataFrame(ret, columns=['date_time', 'long_return'])
    return month_ret


def ts_sum(df, window=10):
    return df.rolling(window).sum()


def annual_ret(net):
    # input daily net
    tot_ret = total_ret(net)
    day = len(net)
    return (tot_ret + 1) ** (365.0 / day) - 1


def AnnualVolatility(net):
    net = net.dropna()
    net_values = net.values
    logreturns = np.diff(np.log(net_values))
    annualVolatility = np.std(logreturns) / np.mean(logreturns)
    annualVolatility = annualVolatility / np.sqrt(1 / 365)
    return annualVolatility


def sharpe_ratio(net):
    # input daily net
    try:
        ret = np.log(net / net.shift(1))
        sharpe = ret.mean() / ret.std() * 365 ** 0.5
    except:
        sharpe = np.nan
    return sharpe


def max_drawdown(A):
    I = -99999999
    for i in range(len(A)-1):
        a = A[i+1:]
        min_a=min(a)
        maxval = 1-min_a/A[i]
        I = max(I,maxval)
    return I*100


def mkfpath(folder, fname):
    try:
        os.mkdir(folder)
    except:
        pass
    fpath = folder + '/' + fname
    return fpath


def summary_net(net_df, plot_in_loops,alpha):
    month_ret = month_profit(net_df)
    # 转换成日净值
    net_df.set_index('date_time', inplace=True)
    # print(net_df)
    # print(len(net_df))
    net_df = net_df.resample(rule='1D').apply({"net": "last", "index": "last"})
    # net_df.reset_index(inplace=True)
    # print(net_df.asfreq())
    net_df.dropna(how="all", inplace=True)

    # 计算汇总
    net_df["date_time"]=net_df.index
    net = net_df['net']
    tot_ret = total_ret(net)
    ann_ret = annual_ret(net)
    sharpe = sharpe_ratio(net)
    annualVolatility = AnnualVolatility(net)
    drawdown = max_drawdown(net.values)
    ret_r = ann_ret / drawdown

    result = [tot_ret, ann_ret, sharpe, annualVolatility,
              drawdown, ret_r,
              net_df['date_time'].iloc[0], net_df['date_time'].iloc[-1]]
    cols = ['tot_ret', 'ann_ret', 'sharpe', 'annualVolatility', 'max_drawdown', 'ret_ratio', 'start_time', 'end_time']

    if plot_in_loops:
        param_str = alpha

        net_df['index'] = net_df["index"]/net_df["index"].iloc[0]
        net_df['net'] = net_df['net'] / net_df['net'].iloc[0]

        fpath = mkfpath('/Users/wuyong/alldata/original_data/temp/', param_str + '.png')

        fig, ax = plt.subplots(2)
        net_df.plot(x='date_time', y=['index', 'net'], title=param_str, grid=True, ax=ax[0])
        ax[0].set_xlabel('')
        month_ret['month'] = month_ret['date_time'].dt.strftime('%Y-%m')
        month_ret.plot(kind='bar', x='month',
                       y=['long_return'],
                       color=['r'], grid=True, ax=ax[1])
        plt.tight_layout()
        plt.savefig(fpath)
        # plt.show()

    return result, cols


slippage = 0.002

symbols = ["ethbtc", "xrpbtc", "mdabtc", "eosbtc", "xlmbtc", "tusdbtc", "ltcbtc",
           "stratbtc", "trxbtc", "adabtc", "iotabtc", "xmrbtc", "bnbbtc", "dashbtc",
           "xembtc", "etcbtc", "neobtc", "ontbtc", "zecbtc", "wavesbtc", "btgbtc",
           "vetbtc", "qtumbtc", "omgbtc", "zrxbtc", "gvtbtc", "bchabcbtc", "bchsvbtc"]


# symbols = sorted(symbols)

aim_symbols = ["ethbtc", "eosbtc", "xrpbtc", "trxbtc", "tusdbtc", "bchabcbtc", "bchsvbtc", "ontbtc", "ltcbtc", "adabtc", "bnbbtc"]

symbols_close = [x+"_close" for x in ["ethbtc", "eosbtc", "xrpbtc", "trxbtc", "tusdbtc", "bchabcbtc", "bchsvbtc", "ontbtc", "ltcbtc", "adabtc", "bnbbtc"]]


"""
下面这个函数主要用来计算所有币对的多因子综合因子值
"""


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

                if symbol == symbols[0]:
                    df = pd.DataFrame({symbol: data[alpha_list[i]].values, symbol + "_close": data["close"].values,
                                       symbol + "_open": data["open"].values, "tickid": data["tickid"].values},
                                      index=data["date"].values)
                else:
                    df_1 = pd.DataFrame({symbol: data[alpha_list[i]].values, symbol + "_close": data["close"].values,
                                         symbol + "_open": data["open"].values}, index=data["date"].values)
                    df = df.merge(df_1, how="left", left_index=True, right_index=True)
            df_symbols = df[symbols]
            df_symbols = df_symbols.rank(axis=1, numeric_only=True,na_option="keep")
            if i == 0:
                df_symbols_last = df_symbols
            else:
                df_symbols_temp = df_symbols
                df_symbols_last = df_symbols_last+df_symbols_temp

        except (FileNotFoundError, TypeError) as e:
            print(e)
    df[symbols] = df_symbols_last
    return df


"""
下面这个函数主要是用于止损，止损的逻辑是出现净值回撤超过10%则止损，所谓净值回撤超过10%是指，买入某个币对（如ETHBTC）之后，随着买入的币对的涨跌，策略当前净值相对于
买入该币对之后的最大净值下跌了10%以上，则直接平仓所持有的币对（如ETHBTC）。当然，加上止损逻辑之后策略的表现并没有提升，因此后面的策略框架当中并没有加上这个逻辑。
"""


def stop_loss(hold_symbles_list, coinnum_dict_now, price_values, n=1, max_price=0, cash_now=0):
    if n == len(price_values):
        return hold_symbles_list, coinnum_dict_now, max_price, cash_now

    elif (max_price-price_values[n])/max_price >= 0.1:
        cash_now = coinnum_dict_now[hold_symbles_list[0]]*price_values[n]*(1-slippage)
        coinnum_dict_now[hold_symbles_list[0]] = 0
        hold_symbles_list = []
        max_price = 0
        print("10%止损")
        return hold_symbles_list, coinnum_dict_now, max_price, cash_now

    else:
        if max_price < max(price_values[n], price_values[n-1]):
            max_price = max(price_values[n], price_values[n-1])
        else:
            pass
        return stop_loss(hold_symbles_list, coinnum_dict_now, price_values, n+1, max_price, cash_now)


"""
下面是多因子策略的交易逻辑
"""


@tail_call_optimized
def multiple_factor(df,cash_list=[10000],asset_list=[10000],buy_list=[[]],coinnum_list=[{}],n=1,close_list=[[]],max_price_list=[0],posittion=1):
    if n == len(df):
        return cash_list, asset_list, buy_list, coinnum_list, close_list, max_price_list

    df_last_all = df.ix[n - 1]
    df_last_min = df[symbols].ix[n - 1]
    max_symbols_list = [df_last_min.idxmax()]
    hold_symbols_list = buy_list[-1]
    aim_symbols_list = list(set(max_symbols_list).intersection(set(aim_symbols)))
    strong_symbols_list = []

    for y in range(len(aim_symbols_list)):
        if df_last_all[aim_symbols_list[y]+"_close"] > df_last_all[aim_symbols_list[y]+"_close25"]:
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

    if len(hold_symbols_list) > 0 and False:
        if hold_symbols_list == buy_list[-1]:
            startdate = df_now_all["date_time"]
            startdate = datetime.datetime.strptime(startdate, "%Y-%m-%d %H:%M:%S")
            time_delta = datetime.timedelta(days=1)
            time_delta1 = datetime.timedelta(hours=4)
            enddate = startdate+time_delta
            enddate1 = startdate+time_delta1
            errcode, errmsg, df_4h_1m = get_exsymbol_kline("BIAN", hold_symbols_list[0], "1m", str(startdate)[:10], str(enddate)[:10])
            price_values = df_4h_1m[(df_4h_1m["date"] > str(startdate)) & (df_4h_1m["date"] < str(enddate1))]["open"].values
            hold_symbols_list, coinnum_dict_now, max_price, cash_now = stop_loss(hold_symbols_list, coinnum_dict_now, price_values, n=1, max_price=max_price_list[-1], cash_now=cash_now)
        else:
            startdate = df_now_all["date_time"]
            startdate = datetime.datetime.strptime(startdate, "%Y-%m-%d %H:%M:%S")
            time_delta = datetime.timedelta(days=1)
            time_delta1 = datetime.timedelta(hours=4)
            enddate = startdate + time_delta
            enddate1 = startdate + time_delta1
            errcode, errmsg, df_4h_1m = get_exsymbol_kline("BIAN", hold_symbols_list[0], "1m", str(startdate)[:10], str(enddate)[:10])
            price_values = df_4h_1m[(df_4h_1m["date"] > str(startdate)) & (df_4h_1m["date"] < str(enddate1))]["open"].values
            hold_symbols_list, coinnum_dict_now, max_price, cash_now = stop_loss(hold_symbols_list, coinnum_dict_now, price_values, n=1, max_price=0, cash_now=cash_now)
        max_price_list.append(max_price)
    else:
        pass

    # 更新买入标的列表和买入数量列表
    buy_list.append(hold_symbols_list)
    coinnum_list.append(coinnum_dict_now)


    cash_list.append(cash_now)
    asset = cash_list[-1]
    for b in range(len(hold_symbols_list)):
        asset = asset + coinnum_dict_now[hold_symbols_list[b]]*df_now_all[hold_symbols_list[b] + "_close"]
    asset_list.append(asset)
    return multiple_factor(df,cash_list, asset_list, buy_list, coinnum_list, n+1, close_list, max_price_list, posittion)

#
# alpha_two_combine = [("Alpha.alpha003", "Alpha.alpha014", "Alpha.alpha050","Alpha.alpha051",
#                       "Alpha.alpha069", "Alpha.alpha128","Alpha.alpha167", "Alpha.alpha175")]
# alpha_two_combine = [("Alpha.alpha003", "Alpha.alpha014", "Alpha.alpha028", "Alpha.alpha050","Alpha.alpha051",
#                       "Alpha.alpha069", "Alpha.alpha096", "Alpha.alpha128","Alpha.alpha167", "Alpha.alpha175")]
# alpha_two_combine = [("Alpha.alpha003","Alpha.alpha024","Alpha.alpha051", "Alpha.alpha052", "Alpha.alpha069",
#                       "Alpha.alpha159", "Alpha.alpha167", "Alpha.alpha175")]
# alpha_two_combine = [("Alpha.alpha069", "Alpha.alpha051", "Alpha.alpha167", "Alpha.alpha175", "Alpha.alpha018")]
alpha_two_combine = [("Alpha.alpha069", "Alpha.alpha018", "Alpha.alpha050", "Alpha.alpha052", "Alpha.alpha055",
                      "Alpha.alpha060", "Alpha.alpha071", "Alpha.alpha052")]  # 比如，这是其中一种多因子组合
df = pick_coin(alpha_two_combine[0])  # 把这种因子组合放入到计算所有币对因子值的函数当中，这里多种因子的因子值计算的方式是打分法的方式，简单来说加总的不是各个币对的因子值而是币对的因子排名

for close in symbols_close:  # 这部分主要用于计算币对的25日均价
    try:
        df[close + str(25)] = ta.MA(df[close].values, timeperiod=25, matype=0)
    except :
        pass

df["date_time"] = df.index.values
print(df.head())
cash_list, asset_list, buy_list, coinnum_list, close_list, max_price_list = multiple_factor(df)  # 这个地方计算出某一因子组合下策略运行之后的现金、净值的时间序列
df_result = pd.DataFrame({"cash": cash_list, "asset": asset_list, "tickid": df["tickid"].values, "buy": buy_list}, index=df["date_time"])
print(df_result.tail())
exit()

"""
下面这个部分主要是画图，把策略净值变化和BTCUSDT的日线级别行情图放到一起，可以用来查看BTCUSDT行情变化对策略净值的影响
"""


df_result["date_time"] = pd.to_datetime(df_result.index.values)
df_result_day = df_result.resample(rule="1d", on='date_time', label="left").apply({"asset": "last", "tickid": "last"})
df_result_day["tickid"] = df_result_day["tickid"] - 14400*5
# print(df_result_day.head())
# print(len(df_result_day))
data_k = pd.read_csv("/Users/wuyong/alldata/original_data/btcusdt_day_k.csv", index_col=0)
# print(len(data_k))
data_k = data_k.merge(df_result_day, on="tickid", how="left")
data_k = data_k[data_k["asset"] > 0]
data_k.fillna(inplace=True, method="ffill")
data_k["ma20"] = ta.EMA(data_k["close"].values, 20)
data_k["ma60"] = ta.EMA(data_k["close"].values, 60)
data_k["ma120"] = ta.EMA(data_k["close"].values, 120)
candleData = np.column_stack([list(range(len(data_k))), data_k[["open", "high", "low", "close"]]])
fig = plt.figure(figsize=(30, 12))
ax = fig.add_axes([0.1, 0.3, 0.8, 0.6])
mpf.candlestick_ohlc(ax, candleData, width=0.5, colorup='r', colordown='b')
lns1 = ax.plot(data_k["date"].values, data_k["ma20"], label="ma20")
lns2 = ax.plot(data_k["date"].values, data_k["ma60"], label="ma60")
lns3 = ax.plot(data_k["date"].values, data_k["ma120"], label="ma120")
plt.grid(True)
plt.xticks(list(range(len(data_k))), list(data_k["date"].values))
plt.xticks(rotation=85)
plt.tick_params(labelsize=5)
ax2 = ax.twinx()
lns4 = ax2.plot(data_k["date"].values, data_k["asset"], label="asset", color="#8B0000")
lns = lns1 + lns2 + lns3 + lns4
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc=0)
plt.show()

exit()

"""
下面这个部分可以用来一次性测试多种多因子组合
"""


dataf = pd.read_csv('/Users/wuyong/alldata/factor_writedb/factor_stra_4h/BIAN_' + "tusdbtc" + "_" + "Alpha.alpha024" + "_gtja4h" + '.csv', index_col=0)
print(dataf.tail())
dataf = dataf[dataf["tickid"] > 1530093600]
dataf.drop_duplicates(subset="tickid", keep="last", inplace=True)
index_values = list(dataf["close"].values)

alpha_two_combine = [("Alpha.alpha069", "Alpha.alpha018", "Alpha.alpha050", "Alpha.alpha052", "Alpha.alpha055",
                      "Alpha.alpha060", "Alpha.alpha071", "Alpha.alpha052")]
#
stat_ls = []
for n in range(len(alpha_two_combine)):
    df = pick_coin(alpha_two_combine[n])
    alpha_name = ""
    for name in alpha_two_combine[n]:
        alpha_name += name[-3:]

    for close in symbols_close:
        try:
            df[close + str(25)] = ta.MA(df[close].values, timeperiod=25, matype=0)
        except :
            pass

    df["date_time"] = df.index.values

    cash_list, asset_list, buy_list, coinnum_list, close_list, max_price_list = multiple_factor(df,cash_list=[10000],asset_list=[10000],buy_list=[[]],coinnum_list=[{}],n=1,close_list=[[]],max_price_list=[0],posittion=1)
    df_result = pd.DataFrame({"cash": cash_list, "asset": asset_list, "tickid": df["tickid"].values, "buy": buy_list}, index=df["date_time"])
    df_result["asset_diff"] = df_result["asset"].diff()
    df_result["date_time"] = pd.to_datetime(df_result.index)
    df_result["date_time"] = df_result.index
    df_result.index = range(len(df_result))
    df_result["net"] = df_result["asset"]
    # print(df_result)
    df_result["index"] = index_values
    df_result["close"] = index_values
    df_result["date_time"] = pd.to_datetime(df_result["date_time"])
    # print(df_result[["net","asset_diff","buy","asset","date_time"]])
    result, cols = summary_net(df_result[["net", "close", "index", "date_time"]], 0, "multifactor" + alpha_name)
    result = [alpha_name] + result
    cols = ["alpha"] + cols
    stat_ls.append(result)
    print(alpha_two_combine[n])
    print(df_result.iloc[-1, :])
    print("\n")

df_last = pd.DataFrame(stat_ls, columns=cols)
df_last = df_last.sort_values(by="ret_ratio", ascending=False)
print(df_last)








































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
from lib.myfun import *
import copy
import pandas as pd
import numpy as np
import talib as ta
import copy
import matplotlib.pyplot as plt
from lib.mutilple_factor_test import *

#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)

def psar(barsdata, iaf=0.02, maxaf=0.2):
    length = len(barsdata)
    dates = list(barsdata['date'])
    high = list(barsdata['high'])
    low = list(barsdata['low'])
    close = list(barsdata['close'])
    psar = close[0:len(close)]
    psarbull = [None] * length
    psarbear = [None] * length
    bull = True
    af = iaf
    hp = high[0]
    lp = low[0]

    for i in range(2, length):
        if bull:
            psar[i] = psar[i - 1] + af * (hp - psar[i - 1])
        else:
            psar[i] = psar[i - 1] + af * (lp - psar[i - 1])

        reverse = False

        if bull:
            if low[i] < psar[i]:
                bull = False
                reverse = True
                psar[i] = hp
                lp = low[i]
                af = iaf
        else:
            if high[i] > psar[i]:
                bull = True
                reverse = True
                psar[i] = lp
                hp = high[i]
                af = iaf

        if not reverse:
            if bull:
                if high[i] > hp:
                    hp = high[i]
                    af = min(af + iaf, maxaf)
                if low[i - 1] < psar[i]:
                    psar[i] = low[i - 1]
                if low[i - 2] < psar[i]:
                    psar[i] = low[i - 2]
            else:
                if low[i] < lp:
                    lp = low[i]
                    af = min(af + iaf, maxaf)
                if high[i - 1] > psar[i]:
                    psar[i] = high[i - 1]
                if high[i - 2] > psar[i]:
                    psar[i] = high[i - 2]

        if bull:
            psarbull[i] = psar[i]
        else:
            psarbear[i] = psar[i]

    return {"dates": dates, "high": high, "low": low, "close": close, "psar": psar, "psarbear": psarbear,
            "psarbull": psarbull}

exchange = 'BIAN'
symbols = ['eosusdt']

dataf = read_data(exchange, symbols[0], '1h', "2017-01-01", "2018-10-01")

def ts_sum(df ,window=10):
    return df.rolling(window).sum()

def GarmanKlass_Vol_1(data,window=30):
    a = 0.5 * np.log(data['high'] / data['low']) ** 2
    b = (2 * np.log(2) - 1) * (np.log(data['close'] / data['open']) ** 2)
    vol_mid1=a-b
    vol_mid1=ts_sum(vol_mid1,window)/window
    return np.sqrt(vol_mid1)*100

dataf["vol"]=GarmanKlass_Vol_1(dataf)
result=psar(dataf)
result=pd.DataFrame.from_dict(result)


def adx(stock_data,n=14):
    df=copy.deepcopy(stock_data)

    df["hd"]=df["high"]-df["high"].shift(1)
    df["ld"]=df["low"].shift(1)-df["low"]

    df["t1"]=df["high"]-df["low"]
    df["t2"]=abs(df["high"]-df["close"].shift(1))
    df.ix[df["t1"]>=df["t2"],"temp1"]=df["t1"]
    df.ix[df["t1"]<df["t2"],"temp1"]=df["t2"]

    df["temp2"]=abs(df["low"]-df["close"].shift(1))

    df.ix[df["temp1"]>=df["temp2"],"temp"]=df["temp1"]
    df.ix[df["temp1"] < df["temp2"], "temp"] = df["temp2"]

    df.dropna(inplace=True)

    df["tr"] = df["temp"].rolling(window=n).sum()

    df.ix[(df["hd"]>0) & (df["hd"]>df["ld"]),"hd1"]=df["hd"]
    df["hd1"].fillna(0,inplace=True)

    df.ix[(df["ld"]>0) & (df["ld"]>df["hd"]),"ld1"]=df["ld"]
    df["ld1"].fillna(0,inplace=True)

    df["dmp"]=df["hd1"].rolling(window=n).sum()
    df["dmm"]=df["ld1"].rolling(window=n).sum()

    df["pdi"]=df["dmp"]/df["tr"]*100
    df["mdi"]=df["dmm"]/df["tr"]*100

    df.ix[df["pdi"]>df["mdi"],"signal"]=1
    df.ix[df["pdi"]<df["mdi"],"signal"]=-1

    df["dx"]=abs(df["pdi"]-df["mdi"])/(df["pdi"]+df["mdi"])*100
    df["adx"]=df["dx"].rolling(window=n).mean()

    df["signal"].fillna(method="ffill",inplace=True)

    return df

#print(adx(dataf).tail())
result_adx=adx(dataf)
result_adx["psarbull"]=result["psarbull"]
result_adx["psarbear"]=result["psarbear"]
print(result_adx.head(40))
print(len(result_adx))

@tail_call_optimized
def saradx_long(cash_list,btcnum_list,fee_list,pos_list):
    fee = 0.00075
    slippage = 0.00025
    if len(btcnum_list)>=len(result_adx):
        print(len(cash_list))
        return cash_list,btcnum_list,fee_list,pos_list

    try:
        df_got = result_adx.ix[:28+len(btcnum_list)]
        df_now=result_adx.ix[29+len(cash_list)]
        adx_value = df_got["adx"].values[-1]
        psar_bull = df_got["psarbull"].values[-1]
        psar_bear = df_got["psarbear"].values[-1]
        vol=df_now.vol

        if adx_value>0:
            if btcnum_list[-1]==0:
                recent_bull = df_got["psarbull"].values[-10:]
                recent_bear = df_got["psarbear"].values[-30:]
                if (psar_bull > 0) and (vol < 4) and (len(recent_bull[recent_bull>0])>6) :
                    buy_price = df_now.open * (1 + slippage)
                    buyamount = cash_list[-1] / buy_price
                    btcnum_list.append(buyamount)
                    cash_list.append(0)
                    fee_list.append(buy_price * buyamount * fee)
                    pos_list.append(1)
                    return saradx_long(cash_list, btcnum_list, fee_list, pos_list)
                else:
                    btcnum_list.append(btcnum_list[-1])
                    cash_list.append(cash_list[-1])
                    fee_list.append(0)
                    pos_list.append(pos_list[-1])
                    return saradx_long(cash_list, btcnum_list, fee_list, pos_list)
            else:
                recent_bear = df_got["psarbear"].values[-10:]
                if psar_bull > 0 and vol < 4:
                    btcnum_list.append(btcnum_list[-1])
                    cash_list.append(cash_list[-1])
                    fee_list.append(0)
                    pos_list.append(pos_list[-1])
                    return saradx_long(cash_list, btcnum_list, fee_list, pos_list)

                elif ((psar_bear>0) and (vol<4) and (adx_value>25) and (len(recent_bear[recent_bear>0])<=6)):
                    btcnum_list.append(btcnum_list[-1])
                    cash_list.append(cash_list[-1])
                    fee_list.append(0)
                    pos_list.append(pos_list[-1])
                    return saradx_long(cash_list, btcnum_list, fee_list, pos_list)

                else:
                    sell_price = df_now.open * (1 - slippage)
                    sellamount = btcnum_list[-1]
                    btcnum_list.append(0)
                    cash_list.append(cash_list[-1] + sellamount * sell_price)
                    fee_list.append(sell_price * sellamount * fee)
                    pos_list.append(0)
                    return saradx_long(cash_list, btcnum_list, fee_list, pos_list)


        else:
            cash_list.append(cash_list[-1])
            btcnum_list.append(btcnum_list[-1])
            fee_list.append(0)
            pos_list.append(pos_list[-1])
            return saradx_long(cash_list,btcnum_list,fee_list,pos_list)
    except IndexError:
        cash_list.append(10)
        btcnum_list.append(0)
        fee_list.append(0)
        pos_list.append(0)
        return saradx_long(cash_list, btcnum_list, fee_list, pos_list)

cash_list,btcnum_list,fee_list,pos_list=saradx_long([],[],[],[])
df=pd.DataFrame({"cash":cash_list,"btc":btcnum_list,"fee":fee_list,"pos":pos_list})
df[["date","close","vol","high","low"]]=result_adx[["date","close","vol","high","low"]]
df.to_csv("saradx_long_eos.csv")
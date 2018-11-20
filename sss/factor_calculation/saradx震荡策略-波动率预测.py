# coding=utf-8

import sys
sys.path.append('..')
from lib.myfun import *
import copy
import pandas as pd
import numpy as np
import talib as ta
import copy
from sklearn.preprocessing import StandardScaler
from lib.mutilple_factor_test import *
import math
from statsmodels.tsa.ar_model import AR
import matplotlib.pyplot as plt

ss = StandardScaler()

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


exchange = 'BIAN'
symbols = ['btcusdt']

ROLLING_WINDOW = 120  # 用到的滚动周期
VOL_NAME = 'GarmanKlassVol'  #计算的波动率名字，几个波动率都可以试一下，YangZhangVol最常用

dataf = read_data(exchange, symbols[0], '1h', "2017-01-01", "2018-10-01")
print(dataf.head())

def GarmanKlass_get_estimator(data,window=30):
    a = 0.5 * np.log(data['high'] / data['low']) ** 2
    b = (2 * np.log(2) - 1) * (np.log(data['close'] / data['open']) ** 2)
    vol_mid1=a-b
    vol_mid1=ts_sum(vol_mid1,window)/window
    return np.sqrt(vol_mid1)*100

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
    vols_pred = df["vol"].rolling(window).apply(fit)
    vols_pred.name = 'AR' + '_' + repr(window) + '_' + repr(p)
    print(vols_pred.name + " prediction finished.")
    return vols_pred

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


dataf["vol"]=GarmanKlass_get_estimator(dataf)
print(dataf.head(40))


#vols_pred_BM=predict_BM(df)

vols_pred_AR=predict_AR(dataf)
dataf["vol_pre"]=vols_pred_AR

result=psar(dataf)
result=pd.DataFrame.from_dict(result)
result_adx=adx(dataf)
result_adx["psarbull"]=result["psarbull"]
result_adx["psarbear"]=result["psarbear"]
print(result_adx.head())


cash_list=[]
btcnum_list=[]
signal_list=[]
fee_list=[]
pos_list=[]
fee = 0.00075
slippage = 0.00025


for rid,row in result_adx.iterrows():
    df_temp=result_adx.iloc[:rid-149,:]
    try:
        adx_value=df_temp["adx"].values[-1]
        psar_bull=df_temp["psarbull"].values[-1]
        psar_bear=df_temp["psarbear"].values[-1]

        if btcnum_list[-1]==0 and adx_value>0:
            if psar_bull>0 and row.vol_pre<2.5:
                buy_price=row.open*(1+slippage)
                buyamount=cash_list[-1]/buy_price
                btcnum_list.append(buyamount)
                cash_list.append(0)
                fee_list.append(buy_price*buyamount*fee)
                signal_list.append("b")
                pos_list.append(1)
            elif psar_bear>0 and row.vol_pre<2.5:
                sell_price=row.open*(1-slippage)
                sellamount=cash_list[-1]/(2*sell_price)
                btcnum_list.append(-sellamount)
                cash_list.append(cash_list[-1]+sell_price*sellamount)
                fee_list.append(sell_price*sellamount*fee)
                signal_list.append("s")
                pos_list.append(-1)
            else:
                print("xx1")
                btcnum_list.append(btcnum_list[-1])
                signal_list.append(np.nan)
                cash_list.append(cash_list[-1])
                fee_list.append(0)
                pos_list.append(pos_list[-1])

        elif btcnum_list[-1]>0 and adx_value>0:
            if psar_bull>0 and row.vol_pre<2.5:
                btcnum_list.append(btcnum_list[-1])
                signal_list.append(np.nan)
                cash_list.append(cash_list[-1])
                fee_list.append(0)
                pos_list.append(pos_list[-1])
            elif psar_bear>0 and row.vol_pre<2.5:
                if adx_value>25:
                    recent_bear=df_temp["psarbear"].values[-10:]
                    if len(recent_bear[recent_bear>0]) > 4:
                        sell_price = row.open * (1 - slippage)
                        sellamount=btcnum_list[-1]
                        cash_now=cash_list[-1]+sell_price*sellamount
                        fee_now=sellamount*sell_price*fee
                        sellamount = cash_now / (2 * sell_price)
                        btcnum_list.append(-sellamount)
                        cash_list.append(cash_list[-1]+cash_now+sellamount*sell_price)
                        fee_list.append(fee_now+sell_price*sellamount*fee)
                        signal_list.append("s")
                        pos_list.append(-1)
                    else :
                        btcnum_list.append(btcnum_list[-1])
                        signal_list.append(np.nan)
                        cash_list.append(cash_list[-1])
                        fee_list.append(0)
                        pos_list.append(pos_list[-1])

                else:
                    sell_price = row.open * (1 - slippage)
                    sellamount = btcnum_list[-1]
                    cash_now = cash_list[-1] + sell_price * sellamount
                    fee_now = sellamount * sell_price*fee
                    sellamount = cash_now / (2 * sell_price)
                    btcnum_list.append(-sellamount)
                    cash_list.append(cash_list[-1] + cash_now + sellamount * sell_price)
                    fee_list.append(fee_now + sell_price * sellamount*fee)
                    signal_list.append("s")
                    pos_list.append(-1)

            else:
                print("xx2")
                sell_price = row.open * (1 - slippage)
                sellamount = btcnum_list[-1]
                btcnum_list.append(0)
                signal_list.append("p")
                cash_list.append(cash_list[-1] + sellamount * sell_price)
                fee_list.append(sell_price * sellamount * fee)
                pos_list.append(0)

        elif btcnum_list[-1]<0 and adx_value>0:
            if psar_bear>0 and row.vol_pre<2.5:
                btcnum_list.append(btcnum_list[-1])
                signal_list.append(np.nan)
                cash_list.append(cash_list[-1])
                fee_list.append(0)
                pos_list.append(pos_list[-1])
            elif psar_bull>0 and row.vol_pre<2.5:
                if adx_value>25:
                    recent_bull = df_temp["psarbull"].values[-10:]
                    if len(recent_bull[recent_bull > 0]) > 4:
                        buy_price = row.open * (1 + slippage)
                        buyamount = abs(btcnum_list[-1])
                        cash_now = cash_list[-1] - buy_price * buyamount
                        fee_now = buyamount * buy_price*fee
                        buyamount = cash_now / buy_price
                        btcnum_list.append(buyamount)
                        cash_list.append(0)
                        fee_list.append(fee_now + buyamount * buy_price*fee)
                        signal_list.append("b")
                        pos_list.append(1)
                    else:
                        btcnum_list.append(btcnum_list[-1])
                        signal_list.append(np.nan)
                        cash_list.append(cash_list[-1])
                        fee_list.append(0)
                        pos_list.append(pos_list[-1])
                else:
                    buy_price = row.open * (1 + slippage)
                    buyamount = abs(btcnum_list[-1])
                    cash_now = cash_list[-1] - buy_price * buyamount
                    fee_now = buy_price * buyamount*fee
                    buyamount = cash_now / buy_price
                    btcnum_list.append(buyamount)
                    cash_list.append(0)
                    fee_list.append(fee_now + buyamount * buy_price*fee)
                    signal_list.append("b")
                    pos_list.append(1)

            else:
                print("xx3")
                buy_price=row.open * (1 + slippage)
                buyamount = abs(btcnum_list[-1])
                cash_now = cash_list[-1] - buy_price * buyamount
                cash_list.append(cash_now)
                btcnum_list.append(0)
                fee_list.append(buy_price*buyamount*fee)
                signal_list.append("p")
                pos_list.append(0)

        else:
            btcnum_list.append(btcnum_list[-1])
            signal_list.append(np.nan)
            cash_list.append(cash_list[-1])
            fee_list.append(0)
            pos_list.append(pos_list[-1])

    except IndexError:
        btcnum_list.append(0)
        signal_list.append(np.nan)
        cash_list.append(100000)
        fee_list.append(0)
        pos_list.append(0)
print(len(cash_list))
print(len(btcnum_list))
print(len(result_adx))
result_adx["cash"]=cash_list
result_adx["btc"]=btcnum_list
result_adx["fee"]=fee_list
result_adx["pos"]=pos_list
result_adx["signal"]=signal_list
result_adx["asset"]=result_adx["cash"]+result_adx["btc"]*result_adx["close"]
#print(result_adx[["asset","close"]])
result_adx.index=result_adx["date"]
print(type(result_adx))
print(result_adx.tail())
result_adx.to_csv("saradx_5.csv")



























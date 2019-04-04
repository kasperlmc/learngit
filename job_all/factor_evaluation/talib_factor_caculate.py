# coding=utf-8

import sys
sys.path.append('..')
import numpy as np
import pandas as pd
import talib as ta
import copy
from lib.myfun import *

#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)


exchange = 'BITFINEX'

symbols = ["ethbtc", "xrpbtc", "mdabtc", "eosbtc", "xlmbtc", "tusdbtc", "ltcbtc",
           "stratbtc", "trxbtc", "adabtc", "iotabtc", "xmrbtc", "bnbbtc", "dashbtc",
           "xembtc", "etcbtc", "neobtc", "ontbtc", "zecbtc", "wavesbtc", "btgbtc",
           "vetbtc", "qtumbtc", "omgbtc", "zrxbtc", "gvtbtc", "bchabcbtc", "bchsvbtc"]


for i in range(0, len(symbols)):
    dataf = pd.read_csv("/Users/wuyong/alldata/original_data/BIAN_" + symbols[i] + "_4h_2018-01-01_2019-02-14.csv", index_col=0)
    # print(symbols[i])
    dataf["open_" + symbols[i]] = dataf["open"]
    dataf["close_" + symbols[i]] = dataf["close"]
    dataf["high_" + symbols[i]] = dataf["high"]
    dataf["low_" + symbols[i]] = dataf["low"]
    dataf["volume_" + symbols[i]] = dataf["volume"]
    dataf["amount_" + symbols[i]] = dataf["amount"]

    if i == 0:
        data_open = dataf[["open_"+symbols[i], "date"]]
        data_close = dataf[["close_" + symbols[i], "date"]]
        data_high = dataf[["high_" + symbols[i], "date"]]
        data_low = dataf[["low_" + symbols[i], "date"]]
        data_volume = dataf[["volume_" + symbols[i], "date"]]
        data_amount = dataf[["amount_" + symbols[i], "date"]]

    else:
        data_open = data_open.merge(dataf[["open_"+symbols[i], "date"]], how="left",on="date")
        data_close = data_close.merge(dataf[["close_"+symbols[i], "date"]], how="left",on="date")
        data_high = data_high.merge(dataf[["high_" + symbols[i], "date"]], how="left", on="date")
        data_low = data_low.merge(dataf[["low_" + symbols[i], "date"]], how="left", on="date")
        data_volume = data_volume.merge(dataf[["volume_" + symbols[i], "date"]], how="left", on="date")
        data_amount = data_amount.merge(dataf[["amount_" + symbols[i], "date"]], how="left", on="date")

col_list = ["ethbtc", "date", "xrpbtc", "mdabtc", "eosbtc", "xlmbtc", "tusdbtc", "ltcbtc",
            "stratbtc", "trxbtc", "adabtc", "iotabtc", "xmrbtc", "bnbbtc", "dashbtc",
            "xembtc", "etcbtc", "neobtc", "ontbtc", "zecbtc", "wavesbtc", "btgbtc",
            "vetbtc", "qtumbtc", "omgbtc", "zrxbtc", "gvtbtc", "bchabcbtc", "bchsvbtc"]

data_open.columns = col_list
data_close.columns = col_list
data_high.columns = col_list
data_low.columns = col_list
data_volume.columns = col_list
data_amount.columns = col_list


data_list = [data_open, data_high, data_low, data_close, data_volume, data_amount]


class Alphas(object):
    def __init__(self, pn_data):
        """
        :传入参数 pn_data: pandas.Panel
        """
        # 获取历史数据
        self.date = pn_data[0]["date"].values
        self.open = pn_data[0]
        del self.open["date"]
        self.high = pn_data[1]
        del self.high["date"]
        self.low = pn_data[2]
        del self.low["date"]
        self.close = pn_data[3]
        del self.close["date"]
        self.volume = pn_data[4]
        del self.volume["date"]
        self.amount = pn_data[5]
        del self.amount["date"]
        self.returns = pd.DataFrame((self.close.values/self.close.shift(1).values)-1, columns=self.close.columns)
        self.vwap = pd.DataFrame(self.amount.values/(self.volume.values*100), columns=self.close.columns)

    def talib_001(self):
        data_ma20 = copy.deepcopy(self.close)
        for symbol in symbols:
            data_ma20[symbol] = ta.MA(data_ma20[symbol].values, timeperiod=20)
        return data_ma20

    def talib_002(self):
        data_ema = copy.deepcopy(self.close)
        for symbol in symbols:
            data_ema[symbol] = ta.EMA(data_ema[symbol].values, timeperiod=30)
        return data_ema

    def talib_003(self):
        data_dema = copy.deepcopy(self.close)
        for symbol in symbols:
            data_dema[symbol] = ta.DEMA(data_dema[symbol].values, timeperiod=30)
        return data_dema

    def talib_004(self):
        data_kama = copy.deepcopy(self.close)
        for symbol in symbols:
            data_kama[symbol] = ta.KAMA(data_kama[symbol].values, timeperiod=30)
        return data_kama

    def talib_005(self):
        data_trima = copy.deepcopy(self.close)
        for symbol in symbols:
            data_trima[symbol] = ta.TRIMA(data_trima[symbol].values, timeperiod=30)
        return data_trima

    def talib_006(self):
        data_sar = copy.deepcopy(self.close)
        for symbol in symbols:
            data_sar[symbol] = ta.SAR(self.high[symbol].values, self.low[symbol].values, acceleration=0, maximum=0)
        return data_sar

    def talib_007(self):
        data_adx = copy.deepcopy(self.close)
        for symbol in symbols:
            data_adx[symbol] = ta.ADX(self.high[symbol].values, self.low[symbol].values, self.close[symbol].values, timeperiod=14)
        return data_adx

    def talib_008(self):
        data_adxr = copy.deepcopy(self.close)
        for symbol in symbols:
            data_adxr[symbol] = ta.ADXR(self.high[symbol].values, self.low[symbol].values, self.close[symbol].values, timeperiod=14)
        return data_adxr

    def talib_009(self):
        data_apo = copy.deepcopy(self.close)
        for symbol in symbols:
            data_apo[symbol] = ta.APO(self.close[symbol].values, fastperiod=12, slowperiod=26, matype=0)
        return data_apo

    def talib_010(self):
        data_AROONOSC = copy.deepcopy(self.close)
        for symbol in symbols:
            data_AROONOSC[symbol] = ta.AROONOSC(self.high[symbol].values, self.low[symbol].values, timeperiod=14)
        return data_AROONOSC

    def talib_011(self):
        data_CCI = copy.deepcopy(self.close)
        for symbol in symbols:
            data_CCI[symbol] = ta.CCI(self.high[symbol].values, self.low[symbol].values, self.close[symbol].values, timeperiod=14)
        return data_CCI

    def talib_012(self):
        data_CMO = copy.deepcopy(self.close)
        for symbol in symbols:
            data_CMO[symbol] = ta.CMO(self.close[symbol].values, timeperiod=14)
        return data_CMO

    def talib_013(self):
        data_DX = copy.deepcopy(self.close)
        for symbol in symbols:
            data_DX[symbol] = ta.DX(self.high[symbol].values, self.low[symbol].values, self.close[symbol].values, timeperiod=14)
        return data_DX

    def talib_014(self):
        data_MFI = copy.deepcopy(self.close)
        for symbol in symbols:
            data_MFI[symbol] = ta.MFI(self.high[symbol].values, self.low[symbol].values, self.close[symbol].values, self.volume[symbol].values, timeperiod=14)
        return data_MFI

    def talib_015(self):
        data_MINUS_DI = copy.deepcopy(self.close)
        for symbol in symbols:
            data_MINUS_DI[symbol] = ta.MINUS_DI(self.high[symbol].values, self.low[symbol].values, self.close[symbol].values, timeperiod=14)
        return data_MINUS_DI

    def talib_016(self):
        data_MINUS_DM = copy.deepcopy(self.close)
        for symbol in symbols:
            data_MINUS_DM[symbol] = ta.MINUS_DM(self.high[symbol].values, self.low[symbol].values, timeperiod=14)
        return data_MINUS_DM

    def talib_017(self):
        data_MOM = copy.deepcopy(self.close)
        for symbol in symbols:
            data_MOM[symbol] = ta.MOM(self.close[symbol].values, timeperiod=10)
        return data_MOM

    def talib_018(self):
        data_PLUS_DI = copy.deepcopy(self.close)
        for symbol in symbols:
            data_PLUS_DI[symbol] = ta.PLUS_DI(self.high[symbol].values, self.low[symbol].values, self.close[symbol].values, timeperiod=14)
        return data_PLUS_DI

    def talib_019(self):
        data_PLUS_DM = copy.deepcopy(self.close)
        for symbol in symbols:
            data_PLUS_DM[symbol] = ta.PLUS_DM(self.high[symbol].values, self.low[symbol].values, timeperiod=14)
        return data_PLUS_DM

    def talib_020(self):
        data_PPO = copy.deepcopy(self.close)
        for symbol in symbols:
            data_PPO[symbol] = ta.PPO(self.close[symbol].values, fastperiod=12, slowperiod=26, matype=0)
        return data_PPO

    def talib_021(self):
        data_ROC = copy.deepcopy(self.close)
        for symbol in symbols:
            data_ROC[symbol] = ta.ROC(self.close[symbol].values, timeperiod=10)
        return data_ROC

    def talib_022(self):
        data_ROCP = copy.deepcopy(self.close)
        for symbol in symbols:
            data_ROCP[symbol] = ta.ROCP(self.close[symbol].values, timeperiod=10)
        return data_ROCP

    def talib_023(self):
        data_ROCR = copy.deepcopy(self.close)
        for symbol in symbols:
            data_ROCR[symbol] = ta.ROCR(self.close[symbol].values, timeperiod=10)
        return data_ROCR

    def talib_024(self):
        data_ROCR100 = copy.deepcopy(self.close)
        for symbol in symbols:
            data_ROCR100[symbol] = ta.ROCR100(self.close[symbol].values, timeperiod=10)
        return data_ROCR100

    def talib_025(self):
        data_RSI = copy.deepcopy(self.close)
        for symbol in symbols:
            data_RSI[symbol] = ta.RSI(self.close[symbol].values, timeperiod=14)
        return data_RSI

    def talib_026(self):
        data_TRIX = copy.deepcopy(self.close)
        for symbol in symbols:
            data_TRIX[symbol] = ta.TRIX(self.close[symbol].values, timeperiod=30)
        return data_TRIX

    def talib_027(self):
        data_ULTOSC = copy.deepcopy(self.close)
        for symbol in symbols:
            data_ULTOSC[symbol] = ta.ULTOSC(self.high[symbol].values, self.low[symbol].values, self.close[symbol].values, timeperiod1=7, timeperiod2=14, timeperiod3=28)
        return data_ULTOSC

    def talib_028(self):
        data_WILLR = copy.deepcopy(self.close)
        for symbol in symbols:
            data_WILLR[symbol] = ta.WILLR(self.high[symbol].values, self.low[symbol].values, self.close[symbol].values, timeperiod=14)
        return data_WILLR

    def talib_029(self):
        data_AD = copy.deepcopy(self.close)
        for symbol in symbols:
            data_AD[symbol] = ta.AD(self.high[symbol].values, self.low[symbol].values, self.close[symbol].values, self.volume[symbol].values)
        return data_AD

    def talib_030(self):
        data_ADOSC = copy.deepcopy(self.close)
        for symbol in symbols:
            data_ADOSC[symbol] = ta.ADOSC(self.high[symbol].values, self.low[symbol].values, self.close[symbol].values, self.volume[symbol].values, fastperiod=3, slowperiod=10)
        return data_ADOSC

    def talib_031(self):
        data_OBV = copy.deepcopy(self.close)
        for symbol in symbols:
            data_OBV[symbol] = ta.OBV(self.close[symbol].values, self.volume[symbol].values)
        return data_OBV

    def talib_032(self):
        data_ATR = copy.deepcopy(self.close)
        for symbol in symbols:
            data_ATR[symbol] = ta.ATR(self.high[symbol].values, self.low[symbol].values, self.close[symbol].values, timeperiod=14)
        return data_ATR

    def talib_033(self):
        data_NATR = copy.deepcopy(self.close)
        for symbol in symbols:
            data_NATR[symbol] = ta.NATR(self.high[symbol].values, self.low[symbol].values, self.close[symbol].values, timeperiod=14)
        return data_NATR

    def talib_034(self):
        data_TRANGE = copy.deepcopy(self.close)
        for symbol in symbols:
            data_TRANGE[symbol] = ta.TRANGE(self.high[symbol].values, self.low[symbol].values, self.close[symbol].values)
        return data_TRANGE

    def talib_035(self):
        data_AVGPRICE = copy.deepcopy(self.close)
        for symbol in symbols:
            data_AVGPRICE[symbol] = ta.AVGPRICE(self.open[symbol].values, self.high[symbol].values, self.low[symbol].values, self.close[symbol].values)
        return data_AVGPRICE

    def talib_036(self):
        data_MEDPRICE = copy.deepcopy(self.close)
        for symbol in symbols:
            data_MEDPRICE[symbol] = ta.MEDPRICE(self.high[symbol].values, self.low[symbol].values)
        return data_MEDPRICE

    def talib_037(self):
        data_TYPPRICE = copy.deepcopy(self.close)
        for symbol in symbols:
            data_TYPPRICE[symbol] = ta.TYPPRICE(self.high[symbol].values, self.low[symbol].values, self.close[symbol].values)
        return data_TYPPRICE

    def talib_038(self):
        data_WCLPRICE = copy.deepcopy(self.close)
        for symbol in symbols:
            data_WCLPRICE[symbol] = ta.WCLPRICE(self.high[symbol].values, self.low[symbol].values, self.close[symbol].values)
        return data_WCLPRICE

    def talib_039(self):
        data_CDLXSIDEGAP3METHODS = copy.deepcopy(self.close)
        for symbol in symbols:
            data_CDLXSIDEGAP3METHODS[symbol] = ta.CDLXSIDEGAP3METHODS(self.open[symbol].values, self.high[symbol].values, self.low[symbol].values, self.close[symbol].values)
        return data_CDLXSIDEGAP3METHODS

    def talib_040(self):
        data_CDLRISEFALL3METHODS = copy.deepcopy(self.close)
        for symbol in symbols:
            data_CDLRISEFALL3METHODS[symbol] = ta.CDLRISEFALL3METHODS(self.open[symbol].values, self.high[symbol].values, self.low[symbol].values, self.close[symbol].values)
        return data_CDLRISEFALL3METHODS

    def talib_041(self):
        data_TSF = copy.deepcopy(self.close)
        for symbol in symbols:
            data_TSF[symbol] = ta.TSF(self.close[symbol].values, timeperiod=14)
        return data_TSF


Alpha = Alphas(data_list)
data_result = Alpha.talib_041()
print(data_result.head(80))
print(data_result.tail())
# print(data_result.sum(axis=1))
# print(type(data_result["date"].values[0]))
exit()



























































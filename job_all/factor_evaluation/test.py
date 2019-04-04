# coding=utf-8

import sys
sys.path.append('..')
import pandas as pd
import numpy as np

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)

# data_1 = pd.read_csv("/Users/wuyong/alldata/original_data/BIAN_ETHBTC_4h_2018-06-01_2018-12-27.csv",index_col=0)
# data_2 = pd.read_csv("/Users/wuyong/alldata/original_data/BIAN_XRPBTC_4h_2018-06-01_2018-12-27.csv",index_col=0)
# data_1.drop_duplicates(subset="tickid", keep="last", inplace=True)
# data_2.drop_duplicates(subset="tickid", keep="last", inplace=True)
# data_1["date"] = pd.to_datetime(data_1["tickid"].values,unit="s")
# data_2["date"] = pd.to_datetime(data_2["tickid"].values,unit="s")
# print(data_1[data_1["tickid"] > 1530093600])
# print(data_1.head())
# print(data_1.tail())
# print(data_2.head())
# print(data_2.tail())
#
# eth_list = list(data_1["tickid"].values)
# xrp_list = list(data_2["tickid"].values)
# print(len(eth_list),len(xrp_list))
# print(set(eth_list)-set(xrp_list))
# print(len(set(eth_list)),len(set(xrp_list)))
#
# ret = []
# for i in eth_list:
#     if i not in ret:
#         ret.append(i)
#     else:
#         print(i)
# print(ret)



l1 = [1,2,3,4]
a = l1[0] if len(l1) > 0 else None
print(a)

num = 9681
re_l = []
for i in range(60, num-20, 20):
    re_l.append(i)
print(len(re_l))


coin = "btcusdt"
data = pd.read_csv("/Users/wuyong/alldata/original_data/bitfinex_" + coin + "_1h_all.csv", index_col=0)
print(len(data))
print(data.head(30))
data.fillna(method="ffill", inplace=True)
data["date_time"] = pd.to_datetime(data["date"])


data = data.resample(rule="4h", on='date_time',label="left", closed="left").apply({'open': 'first', 'high': 'max',
                                                                                     'low': 'min', 'close': 'last',
                                                                                     'volume': 'sum', "amount": "sum",
                                                                                     "tickid": "first"})

data["date"] = data.index.values
print(len(data))
print(data.head(10))
data.to_csv("/Users/wuyong/alldata/original_data/bitfinex_btcusdt_4h_all.csv")









# ((TSRANK(DECAYLINEAR(DECAYLINEAR(CORR((CLOSE),VOLUME,10),16),4),5) - RANK(DECAYLINEAR(CORR(VWAP,MEAN(VOLUME,30),4),3)))*-1)

# SMA(CLOSE-DELAY(CLOSE,20),20,1)

# LOG((COUNT(CLOSE/DELAY(CLOSE)-1>((CLOSE/DELAY(CLOSE,19))^(1/20)-1),20)-1)*(SUMIF(((CLOSE/DELAY(C LOSE)-1-(CLOSE/DELAY(CLOSE,19))^(1/20)-1))^2,20,CLOSE/DELAY(CLOSE)-1<(CLOSE/DELAY(CLOSE,19))^(1/20)- 1))/((COUNT((CLOSE/DELAY(CLOSE)-1<(CLOSE/DELAY(CLOSE,19))^(1/20)-1),20))*(SUMIF((CLOSE/DELAY(CLOS E)-1-((CLOSE/DELAY(CLOSE,19))^(1/20)-1))^2,20,CLOSE/DELAY(CLOSE)-1>(CLOSE/DELAY(CLOSE,19))^(1/20)-1))) )
print(369/39958)

# ((RANK(MAX((VWAP - CLOSE), 3)) + RANK(MIN((VWAP - CLOSE), 3))) * RANK(DELTA(VOLUME, 3))) (007)

# RANK(DELTA(((((HIGH + LOW) / 2) * 0.2) + (VWAP * 0.8)), 4) * -1) (008)

# RANK(MAX(((RET < 0) ? STD(RET, 20) : CLOSE)^2, 5)) (010)

# (RANK((OPEN - (SUM(VWAP, 10) / 10)))) * (-1 * (RANK(ABS((CLOSE - VWAP))))) (012)

# (-1 * TSMAX(RANK(CORR(RANK(VOLUME), RANK(VWAP), 5)), 5)) (016)

# RANK((VWAP - MAX(VWAP, 15)))^DELTA(CLOSE, 5) (017)

# ((-1 * RANK((DELTA(CLOSE, 7) * (1 - RANK(DECAYLINEAR((VOLUME / MEAN(VOLUME,20)), 9)))))) * (1 + RANK(SUM(RET, 250)))) (025)

# (MIN(RANK(DECAYLINEAR(DELTA(OPEN, 1), 15)), RANK(DECAYLINEAR(CORR((VOLUME), ((OPEN * 0.65) + (OPEN *0.35)), 17),7))) * -1) (035)

# RANK(SUM(CORR(RANK(VOLUME), RANK(VWAP), 6), 2)) (036)

# ((RANK(DECAYLINEAR(DELTA((CLOSE), 2),8)) - RANK(DECAYLINEAR(CORR(((VWAP * 0.3) + (OPEN * 0.7)), SUM(MEAN(VOLUME,180), 37), 14), 12))) * -1)  (039)

# (RANK(MAX(DELTA((VWAP), 3), 5)) * -1)  (041)

# (TSRANK(DECAYLINEAR(CORR(((LOW)), MEAN(VOLUME,10), 7), 6),4) + TSRANK(DECAYLINEAR(DELTA((VWAP), 3), 10), 15))  (044)

# (RANK(DELTA((((CLOSE * 0.6) + (OPEN *0.4))), 1)) * RANK(CORR(VWAP, MEAN(VOLUME,150), 15)))  (045)

# (MAX(RANK(DECAYLINEAR(DELTA(VWAP, 1), 12)), RANK(DECAYLINEAR(RANK(CORR((LOW),MEAN(VOLUME,80), 8)), 17))) * -1)  (061)

# (MAX(RANK(DECAYLINEAR(CORR(RANK(VWAP), RANK(VOLUME), 4), 4)), RANK(DECAYLINEAR(MAX(CORR(RANK(CLOSE), RANK(MEAN(VOLUME,60)), 4), 13), 14))) * -1)  (064)

# ((TSRANK(DECAYLINEAR(DECAYLINEAR(CORR((CLOSE), VOLUME, 10), 16), 4), 5)-RANK(DECAYLINEAR(CORR(VWAP, MEAN(VOLUME,30), 4),3))) * -1)  (073)

# (RANK(CORR(SUM(((LOW * 0.35) + (VWAP * 0.65)), 20), SUM(MEAN(VOLUME,40), 20), 7)) + RANK(CORR(RANK(VWAP), RANK(VOLUME), 6)))  (074)

# MIN(RANK(DECAYLINEAR(((((HIGH + LOW) / 2) + HIGH) - (VWAP + HIGH)), 20)), RANK(DECAYLINEAR(CORR(((HIGH + LOW) / 2), MEAN(VOLUME,40), 3), 6)))  (077)


# ((RANK(DECAYLINEAR(DELTA(VWAP, 4), 7)) + TSRANK(DECAYLINEAR(((((LOW * 0.9) + (LOW * 0.1)) - VWAP) / (OPEN - ((HIGH + LOW) / 2))), 11), 7)) * -1) (087)

# (RANK(CORR(RANK(VWAP), RANK(VOLUME), 5)) * -1)  (090)

# (MAX(RANK(DECAYLINEAR(DELTA(((CLOSE * 0.35) + (VWAP *0.65)), 2), 3)), TSRANK(DECAYLINEAR(ABS(CORR((MEAN(VOLUME,180)), CLOSE, 13)), 5), 15)) * -1)  (092)

# ((RANK((HIGH - MIN(HIGH, 2)))^RANK(CORR((VWAP), (MEAN(VOLUME,120)), 6))) * -1)  (108)

# (RANK(DECAYLINEAR(CORR(VWAP, SUM(MEAN(VOLUME,5), 26), 5), 7)) - RANK(DECAYLINEAR(TSRANK(CORR(RANK(OPEN), RANK(MEAN(VOLUME,15)), 21), 7), 8)))  (119)

# (RANK((VWAP - CLOSE)) / RANK((VWAP + CLOSE))) (120)

# ((RANK((VWAP - MIN(VWAP, 12)))^TSRANK(CORR(TSRANK(VWAP, 20), TSRANK(MEAN(VOLUME,60), 2), 18), 3)) * -1)  (121)

# (CLOSE - VWAP) / DECAYLINEAR(RANK(TSMAX(CLOSE, 30)),2)  (124)

# (RANK(DECAYLINEAR(CORR((VWAP), MEAN(VOLUME,80),17), 20)) / RANK(DECAYLINEAR(DELTA(((CLOSE * 0.5) + (VWAP * 0.5)), 3), 16)))  (125)

# (RANK(DECAYLINEAR(CORR(((HIGH + LOW) / 2), MEAN(VOLUME,40), 9), 10)) / RANK(DECAYLINEAR(CORR(RANK(VWAP), RANK(VOLUME), 7),3)))  (130)

# (RANK(DELAT(VWAP, 1))^TSRANK(CORR(CLOSE,MEAN(VOLUME,50), 18), 18)) (131)

# ((RANK(DECAYLINEAR(DELTA((((LOW * 0.7) + (VWAP *0.3))), 3), 20)) - TSRANK(DECAYLINEAR(TSRANK(CORR(TSRANK(LOW, 8), TSRANK(MEAN(VOLUME,60), 17), 5), 19), 16), 7)) * -1)  (138)

# MIN(RANK(DECAYLINEAR(((RANK(OPEN) + RANK(LOW)) - (RANK(HIGH) + RANK(CLOSE))), 8)), TSRANK(DECAYLINEAR(CORR(TSRANK(CLOSE, 8), TSRANK(MEAN(VOLUME,60), 20), 8), 7), 3))  (140)

# (MAX(RANK(DECAYLINEAR(DELTA(VWAP, 5), 3)), RANK(DECAYLINEAR(((DELTA(((OPEN * 0.15) + (LOW *0.85)), 2) / ((OPEN * 0.15) + (LOW * 0.85))) * -1), 3))) * -1)  (156)

(MIN(PROD(RANK(RANK(LOG(SUM(TSMIN(RANK(RANK((-1 * RANK(DELTA((CLOSE - 1), 5))))), 2), 1)))), 1), 5) + TSRANK(DELAY((-1 * RET), 6), 5))  (157)

# RANK(((((-1 * RET) * MEAN(VOLUME,20)) * VWAP) * (HIGH - CLOSE)))  (163)

# ((((RANK((1 / CLOSE)) * VOLUME) / MEAN(VOLUME,20)) * ((HIGH * RANK((HIGH - CLOSE))) / (SUM(HIGH, 5) / 5))) - RANK((VWAP - DELAY(VWAP, 5))))    (170)

(RANK(CORR(VWAP, VOLUME, 4)) *RANK(CORR(RANK(LOW), RANK(MEAN(VOLUME,50)), 12)))  (179)



























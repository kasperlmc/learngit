import sys
sys.path.append('..')

import pandas as pd
from lib.myfun import *
import numpy as np

# symbol = "ethusdt"
#
# exchange = 'BITFINEX'
#
# dataf = read_data(exchange, symbol, '1h', "2017-01-01", "2018-10-01")
#
# above = dataf.iloc[:350]
# # print(above.tail())
# below = dataf.iloc[350:]
#
# insertRow = dataf.iloc[[349]]
# insertRow.loc[349, "date"] = "2017-01-15 17:00:00"
# insertRow.loc[349,["open","high","low"]] = 9.7147
#
# newData2 = pd.concat([above,insertRow,below],ignore_index=True)
# # print(newData2.head(400))
#
# above = newData2.iloc[:852]
# # print(above.tail())
# below = newData2.iloc[852:]
#
#
# insertRow = newData2.iloc[[851]]
# insertRow.loc[851, "date"] = "2017-02-05 23:00:00"
# insertRow.loc[851,["open", "high", "low"]] = 11.221
#
# newData2 = pd.concat([above,insertRow,below],ignore_index=True)
# print(len(newData2))
# print(newData2.tail())
#
# newData2.to_csv("/Users/wuyong/alldata/original_data/BITFINEX_ethusdt_1h_2017-01-01_2018-10-01.csv")


# symbol = ".bxbt"
#
# exchange = 'BITMEX'
#
# dataf = read_data("BITMEX", ".bxbt", '1h', "2017-01-01", "2018-10-01")
#
# print(dataf.head(20))
# dataf["date_time"] = pd.to_datetime(dataf["date"])
# df = dataf.resample(rule="4h", on='date_time',label="right").apply(
#             {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
#              'volume': 'sum',"amount":"sum"})
# df["index"] = range(len(df))
# df["date"] = df.index
# df.index = df["index"]
#
# print(df)
# print(len(df))
# print(3828-3393)
# df.to_csv("/Users/wuyong/alldata/factor_writedb/factor_stra_4h/BITMEX_.bxbt_4h_2017-01-01_2018-10-01.csv")

a = np.zeros((13,))
print(a)
print(a+1)

combine_value = pd.Series(np.zeros(10),name="mid_value")
combine_value[combine_value>-1]=combine_value+1
print(combine_value)
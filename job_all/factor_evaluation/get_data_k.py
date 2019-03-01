import pandas as pd
import numpy as np

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)


coin_list = ["btcusdt", "ethusdt", "eosusdt", "etcusdt", "xrpusdt"]

open_df = pd.DataFrame()
close_df = pd.DataFrame()
high_df = pd.DataFrame()
low_df = pd.DataFrame()

for coin in coin_list:
    data = pd.read_csv("/Users/wuyong/alldata/original_data/bitfinex_" + coin + "_1h.csv", index_col=0)
    data.index = data["date"].values
    if len(open_df) == 0:
        open_df = data[["open"]]
        open_df.columns = ["open_"+coin]
        close_df = data[["close"]]
        close_df.columns = ["close_" + coin]
        high_df = data[["high"]]
        high_df.columns = ["high_" + coin]
        low_df = data[["low"]]
        low_df.columns = ["low_" + coin]
    else:
        open_df = open_df.merge(data[["open"]], how="left", left_index=True, right_index=True)
        open_df.rename(columns={"open": "open_"+coin}, inplace=True)
        close_df = close_df.merge(data[["close"]], how="left", left_index=True, right_index=True)
        close_df.rename(columns={"close": "close_" + coin}, inplace=True)
        high_df = high_df.merge(data[["high"]], how="left", left_index=True, right_index=True)
        high_df.rename(columns={"high": "high_" + coin}, inplace=True)
        low_df = low_df.merge(data[["low"]], how="left", left_index=True, right_index=True)
        low_df.rename(columns={"low": "low_" + coin}, inplace=True)


high_df.fillna(method="ffill", inplace=True)
open_df.fillna(method="ffill", inplace=True)
low_df.fillna(method="ffill", inplace=True)
close_df.fillna(method="ffill", inplace=True)
date_list = list(close_df.index.values)
test_coin = "btcusdt"
test_k_high = list(high_df["high_"+test_coin].iloc[-60:])
test_k_open = list(open_df["open_"+test_coin].iloc[-60:])
test_k_low = list(low_df["low_"+test_coin].iloc[-60:])
test_k_close = list(close_df["close_"+test_coin].iloc[-60:])


dt = pd.DataFrame(columns=['coin', "startdate", "enddate", "T", "ret"])
y = 0
num = len(high_df)-1
print(num)

for d in range(60, num-20, 20):
    print(d)
    close2 = close_df.iloc[d-59:d+1]
    open2 = open_df.iloc[d - 59:d + 1]
    high2 = high_df.iloc[d - 59:d + 1]
    low2 = low_df.iloc[d - 59:d + 1]
    for coin in coin_list:
        corropen = round(np.corrcoef(test_k_open, open2["open_"+coin])[0][1], 3)
        corrclose = round(np.corrcoef(test_k_close, close2["close_"+coin])[0][1], 3)
        corrhigh = round(np.corrcoef(test_k_high, high2["high_"+coin])[0][1], 3)
        corrlow = round(np.corrcoef(test_k_low, low2["low_"+coin])[0][1], 3)
        return_20 = close_df["close_"+coin].values[d+20]/close_df["close_"+coin].values[d]
        T = (corrclose+corropen+corrhigh+corrlow)/4
        startdate = date_list[d-59]
        enddate = date_list[d+1]
        dt.loc[y] = [coin, startdate, enddate, T, return_20]
        y += 1

dt.fillna(0, inplace=True)
dt.sort_values(by="T", ascending=False, inplace=True)
print(dt.head(20))














































































































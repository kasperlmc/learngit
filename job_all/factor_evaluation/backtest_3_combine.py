import pandas as pd
import numpy as np


# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)

data_all_btc_1h = pd.read_csv("/Users/wuyong/alldata/original_data/bitfinex_btcusdt_1h_all_result.csv", index_col=0)
data_all_btc_4h = pd.read_csv("/Users/wuyong/alldata/original_data/bitfinex_btcusdt_4h_all_result.csv", index_col=0)
print(data_all_btc_4h.head())
data_all_btc_1h.columns = ["asset_1h", "cash_1h", "price_1h", "coinnum_1h"]
print(data_all_btc_1h.head())

data_all = data_all_btc_1h.merge(data_all_btc_4h, how="outer", left_index=True, right_index=True)
print(data_all)


















































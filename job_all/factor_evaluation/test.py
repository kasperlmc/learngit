# coding=utf-8

import sys
sys.path.append('..')
import pandas as pd
import numpy as np

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)

data_1 = pd.read_csv("/Users/wuyong/alldata/original_data/BIAN_ETHBTC_4h_2018-06-01_2018-12-27.csv",index_col=0)
data_2 = pd.read_csv("/Users/wuyong/alldata/original_data/BIAN_XRPBTC_4h_2018-06-01_2018-12-27.csv",index_col=0)
data_1.drop_duplicates(subset="tickid", keep="last", inplace=True)
data_2.drop_duplicates(subset="tickid", keep="last", inplace=True)
data_1["date"] = pd.to_datetime(data_1["tickid"].values,unit="s")
data_2["date"] = pd.to_datetime(data_2["tickid"].values,unit="s")
print(data_1[data_1["tickid"] > 1530093600])
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




































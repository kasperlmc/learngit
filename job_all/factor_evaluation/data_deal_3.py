import sys
sys.path.append('..')

import pandas as pd
from lib.myfun import *
import numpy as np
import time
from datetime import datetime
# 显示所有行
pd.set_option('display.max_rows', None)


def timestamp_to_datetime(timestamp):
    """将 13 位整数的毫秒时间戳转化成本地普通时间格式)
    :param sp: 13 位整数的毫秒时间戳1456402864242)
    :return: 返回格式 {}2016-02-25 20:21:04.242000
    """
    local_dt_time = datetime.fromtimestamp(timestamp/1000.0)
    return local_dt_time


data_results = pd.read_csv("/Users/wuyong/alldata/original_data/trades_BIAN_xrpusdt.csv")
data_results["dealtime"] = data_results["dealtime"].apply(lambda x: int(x/1000))
data_results.drop_duplicates(keep="last",inplace=True,subset="dealtime")
data_last = pd.DataFrame({"dealtime":range(1542094231,1544445061)})
print(data_results.head())
print(data_results.tail())
print(len(data_last),len(data_results))
print(data_results[data_results["dealtime"]>1543680000].head())

# data_last = data_last.merge(data_results,how="left",on="dealtime")
# data_last.fillna(method="ffill",inplace=True)
# print(data_last.head(100))
# data_last.to_csv("/Users/wuyong/alldata/original_data/trades_HUOBI_xrpusdt_s_1.csv")


# symbol = "xrpusdt"
#
# exchange = 'HUOBI'
#
# dataf = read_data(exchange, symbol, '1m', "2018-11-09", "2018-12-11")
# # dataf["tickid"] = dataf["tickid"]+60
# print(dataf.head(10))
# print(dataf.tail())
# # dataf.to_csv("/Users/wuyong/alldata/original_data/trades_bian_eosusdt_m_2.csv")



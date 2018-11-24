# coding=utf-8

import sys
sys.path.append('..')
import pandas as pd
from lib.mutilple_factor_test import *
from backtest import *


#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)

data=pd.read_csv("saradx_long_eos.csv")
data["net"]=data["asset"]-data["fee"]
fee_list=data["fee"]
print(len(fee_list[fee_list>0]))


data_net=data[["net","date_time","pos","close"]]
data_net["date_time"] = pd.to_datetime(data_net["date_time"])
result,cols=summary_net(data_net,0,"saradx_long_eos")
print(result)
stat_ls=[result]
df=pd.DataFrame(stat_ls,columns=cols)
print(df)



'''
month_ret = month_profit(data_net)
long_ret, short_ret = long_short_ret(data_net)
long_hold, short_hold = mean_hold_k(data_net['pos'])
pos = mean_position(data_net['pos'])
long_times, short_times = trade_times(data_net['pos'])
print(month_ret)
print(long_ret,short_ret)
print(long_hold,short_hold)
print(pos)
print(long_times,short_times)


data_net.set_index('date_time', inplace=True)
data_net = data_net.resample('1D').asfreq()
data_net.reset_index(inplace=True)
#print(data_net)
data_net.dropna(inplace=True)
# 计算汇总
net = data_net['net']
#print(net)
date_time = data_net['date_time']
base = data_net['close']
tot_ret = total_ret(net)
ann_ret = annual_ret(date_time, net)
sharpe = sharpe_ratio(date_time, net)
annualVolatility = AnnualVolatility(net)
drawdown = max_drawdown(net.values)
print(tot_ret,ann_ret,sharpe,annualVolatility,drawdown)

'''


'''
print(data.head())
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(data.index,data["vol"])
ax2=ax1.twinx()
ax2.plot(data.index,data["asset"],c="r")
plt.savefig("saradx1.png")
'''





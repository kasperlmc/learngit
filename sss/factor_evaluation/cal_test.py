import sys
sys.path.append('..')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lib.mutilple_factor_test import *
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()

a=list(range(1,192))
alpha_test=[]
for x in a:
    if x<10:
        alpha_test.append("Alpha.alpha00"+str(x))
    elif 10<x<100:
        alpha_test.append("Alpha.alpha0"+str(x))
    else:
        alpha_test.append("Alpha.alpha" + str(x))

alpha_test=[x + "_" + "gtja" for x in alpha_test]

a=list(range(1,61))
for x in a:
    if x<10:
        alpha_test.append("Alpha.alpha00"+str(x))
    else:
        alpha_test.append("Alpha.alpha0"+str(x))


#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)

alpha_test=["Alpha.alpha009","Alpha.alpha118_gtja","Alpha.alpha177_gtja","Alpha.alpha059_gtja"]
corr_list=[1,1,1,-1]
for i in range(len(alpha_test)):
    try:
        factor_df = pd.read_csv('../factor_writedb/btcusdt_'+alpha_test[i]+'.csv', index_col=0)
        params = list(factor_df.columns)[8:]
        for param in params:
            df = calc_alpha_signal(factor_df, param, pd.to_datetime('2017-08-18'), pd.to_datetime('2018-10-01'), corr=corr_list[i])
            net_df, signal_df, end_capital = do_backtest(df, param, pd.to_datetime('2017-08-18'), pd.to_datetime('2018-10-01'))
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            ax1.set_title(param)
            pose_series=net_df["pos"]
            #param_series=factor_df[param]
            #bin_alpha = pd.qcut(param_series, q=10, retbins=True, duplicates="drop")[1]
            close_values=net_df["close"].values
            #param_series[(param_series<bin_alpha[-2])]=np.nan
            index_values=list(net_df.index)
            #print(factor_df[["close",param]])
            ax1.plot(index_values, close_values, '-', label=param)
            ax2 = ax1.twinx()
            ax2.scatter(index_values, pose_series.values,c="r")
            plt.show()
            fig.savefig(param+".png")
    except FileNotFoundError:
        pass














'''
df=factor_df.dropna()
factor=df["Alpha.alpha009_rank_10"]
bin_alpha=pd.qcut(factor,q=10,retbins=True)[1]
print(bin_alpha)

df["open_long_signal"]=(df["Alpha.alpha009_rank_10"].shift(1)>=bin_alpha[-2])*1
df['close_long_signal']=(df["Alpha.alpha009_rank_10"].shift(1)<bin_alpha[-2])*1
df["open_short_signal"]=(df["Alpha.alpha009_rank_10"].shift(1)<=bin_alpha[1])*1
df["close_short_signal"]=(df["Alpha.alpha009_rank_10"].shift(1)>bin_alpha[1])*1

start_day = pd.to_datetime('2018-04-01')
end_day = pd.to_datetime('2018-10-01')
df["date_time"]=pd.to_datetime(df["date"])
df = df[(df['date_time'] >= start_day) & (df['date_time'] < end_day)]
print(df)
'''

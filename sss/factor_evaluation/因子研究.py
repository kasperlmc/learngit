import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)

data=pd.read_csv('btcusdt_'+ "result_df_all" +'_stat.csv', index_col=0)
print(len(data))
'''
data.index=pd.to_datetime(data["date"])
data["delta_c"]=data["close"]-data["close"].shift(1)
trend_list=[1 if x>=0 else -1 for x in data["delta_c"].values]
print(trend_list)
trend_list=np.array(trend_list)
data["trend"]=trend_list
print(data.head())
'''


#data["trend"].plot()
#plt.show()










# coding=utf-8

import sys

sys.path.append('..')
import pandas as pd
import numpy as np

#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)

data=pd.read_csv("double_factor_backtest.csv",index_col=0)
#print(data)
df_last=[]
for i in range(0,len(data),3):
    df_temp=data.ix[i:i+3]
    if len(df_temp[df_temp["sharpe"]>0])==3:
        if len(df_last)==0:
            df_last=df_temp
        else:
            df_last=pd.concat([df_last,df_temp],axis=0)
print(df_last)


































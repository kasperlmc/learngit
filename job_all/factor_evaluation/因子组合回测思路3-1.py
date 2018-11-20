# coding=utf-8

import sys
import datetime
sys.path.append('..')
import pandas as pd
from itertools import combinations
from lib.mutilple_factor_test import *


#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)

data_eva=pd.read_csv("all_xbtusd_factor_eva.csv",index_col=0)


data_eva=data_eva[(data_eva["sharpe_hv"]>1) | (data_eva["sharpe_lv"]>1)]
print(data_eva)
print(len(data_eva))

starttime = datetime.datetime.now()
combins=[0,1,2,5,5,8,9,10,12,13,22,24]+list(range(4,7))+list(range(15,20))
print(combins)
result_list=[]
hv_all=0
lv_all=0
for i in range(len(combins)):
    series_x=data_eva.ix[combins[i]]
    alpha_x=series_x.name
    sharpe_hv_x=series_x.sharpe_hv
    sharpe_lv_x = series_x.sharpe_lv

    hv_all+=sharpe_hv_x
    lv_all+=sharpe_lv_x
result_list.append([hv_all,lv_all])
#print(result_list)
df_result=pd.DataFrame(result_list,columns=["hv_all","lv_all"])
endtime = datetime.datetime.now()
print(df_result)
#print(list(data_eva.index[:25]))
print((endtime - starttime).seconds)

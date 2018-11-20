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

data_eva=pd.read_csv("all_factor_eva.csv",index_col=0)
data_corr=pd.read_csv("corr_all_factor_1.csv",index_col=0)

data_eva=data_eva[(data_eva["sharpe_hv"]>1) | (data_eva["sharpe_lv"]>1)]
#print(data_eva)
combins = [c for c in  combinations(range(len(data_eva)), 2)]
print(len(combins))
starttime = datetime.datetime.now()

result_list=[]
for i in range(len(combins)):
    combin_tuple=combins[i]
    series_x=data_eva.ix[combin_tuple[0]]
    series_y=data_eva.ix[combin_tuple[1]]
    alpha_x=series_x.name
    sharpe_hv_x=series_x.sharpe_hv
    sharpe_lv_x = series_x.sharpe_lv
    alpha_y = series_y.name
    sharpe_hv_y = series_y.sharpe_hv
    sharpe_lv_y = series_y.sharpe_lv
    hv_xy=sharpe_hv_x+sharpe_hv_y
    lv_xy=sharpe_lv_x+sharpe_lv_y
    corr_xy=data_corr.ix[alpha_x][alpha_y]
    temp_list=[alpha_x,alpha_y,sharpe_hv_x,sharpe_lv_x,sharpe_hv_y,sharpe_lv_y,hv_xy,lv_xy,corr_xy]
    result_list.append(temp_list)
#print(result_list)
df_result=pd.DataFrame(result_list,columns=["alpha_x","alpha_y","sharpe_hv_x","sharpe_lv_x","sharpe_hv_y","sharpe_lv_y","hv_xy","lv_xy","corr_xy"])
endtime = datetime.datetime.now()
print((endtime - starttime).seconds)
#print(df_result[(df_result["hv_xy"]>1) & (df_result["lv_xy"]>1) & (df_result["corr_xy"].abs()>0.5)])
#print(len(df_result[(df_result["hv_xy"]>1) & (df_result["lv_xy"]>1) & (df_result["corr_xy"].abs()>0.5)]))
#df_result.to_csv("conbime_sharpe_factors.csv")
#df_filter=df_result[(df_result["hv_xy"]>1) & (df_result["lv_xy"]>1) & (df_result["corr_xy"].abs()>0.5)]

filter_list=[1 if (x[x.rfind("_")+1:]==y[y.rfind("_")+1:] and z>0) or (x[x.rfind("_")+1:]!=y[y.rfind("_")+1:] and z<0) else 0 for x,y,z in zip(df_result["alpha_x"].values,df_result["alpha_y"].values,df_result["corr_xy"].values)]
filter_list=pd.Series(filter_list,index=df_result.index,name="filter")
df_filter=df_result[filter_list==0]
#print(df_filter)
print(df_filter[(df_filter["hv_xy"]>1) & (df_filter["lv_xy"]>1) & (df_filter["corr_xy"].abs()>0.4)])





































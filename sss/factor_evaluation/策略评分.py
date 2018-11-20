import sys
sys.path.append('..')
from backtest import *
from lib import dataapi
import pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

ss = StandardScaler()

#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)

data=pd.read_csv('btcusdt_'+ "result_df_all_all" +'_stat.csv', index_col=0)
data = data.replace([-np.inf, np.inf], 0).fillna(value=0)
eva_columns=["ann_ret", 'max_drawdown', 'annualVolatility', 'win_r', "sharpe"]
data_eva=data
data_eva.index=data_eva["param"]

data_eva=data_eva[eva_columns]

data_eva["max_drawdown_inverse"]=1/data_eva['max_drawdown']
data_eva['annualVolatility_inverse']=1/data_eva['annualVolatility']
data_eva = data_eva.replace([-np.inf, np.inf], 0).fillna(value=0)
data_eva[["ann_ret",'max_drawdown_inverse','annualVolatility_inverse','win_r']]=ss.fit_transform(data_eva[["ann_ret",'max_drawdown_inverse','annualVolatility_inverse','win_r']])
data_eva["sum"]=data_eva[["ann_ret",'max_drawdown_inverse','annualVolatility_inverse','win_r']].apply(lambda x: x.sum(), axis=1)

#print(data_eva.sort_values(by="sum",ascending=False).head())
#print(data_eva.ix[["Alpha.alpha009_rank_10_positive_hv",'Alpha.alpha009_rank_10_positive_lv']])
#print(data_eva.ix[["Alpha.alpha009_rank_10_positive_hv",'Alpha.alpha009_rank_10_positive_lv']]['sum'].sum())

param_list=list(data_eva.index)
#print(param_list)
params_filter=[s[:s.rfind("_")] for s in param_list]
#print(params_filter)
params_filter=list(set(params_filter))

'''
sum_all_params=[data_eva.ix[[x+"_hv",x+'_lv']]['sum'].sum() for x in params_filter]
#df=pd.Series(sum_all_params,index=params_filter,name="sum_hvlv")
#print(df.sort_values(ascending=False))
sum_hv=[data_eva.ix[x+'_hv']['sum'] for x in params_filter]
sum_lv=[data_eva.ix[x+'_lv']['sum'] for x in params_filter]
df=pd.DataFrame({'sum_hvlv':sum_all_params,'sum_hv':sum_hv,'sum_lv':sum_lv},index=params_filter)
#print(df.sort_values(by='sum_hvlv',ascending=False))
'''


column_list=["sum_hvlv",'sum_hv','sum_lv','ann_ret_hv','ann_ret_lv','win_r_hv','win_r_lv','sharpe_hv','sharpe_lv','max_drawdown_hv','max_drawdown_lv','annualVolatility_hv','annualVolatility_lv']
value_list=[]
for x in params_filter:
    sum_hvlv=data_eva.ix[[x+"_hv",x+'_lv']]['sum'].sum()
    sum_hv=data_eva.ix[x+'_hv']['sum']
    sum_lv=data_eva.ix[x+'_lv']['sum']
    ann_ret_hv=data_eva.ix[x+'_hv']['ann_ret']
    ann_ret_lv = data_eva.ix[x + '_lv']['ann_ret']
    win_r_hv=data_eva.ix[x+'_hv']['win_r']
    win_r_lv = data_eva.ix[x + '_lv']['win_r']
    sharp_hv=data_eva.ix[x + '_hv']['sharpe']
    sharp_lv = data_eva.ix[x + '_lv']['sharpe']
    max_drawdown_hv=data_eva.ix[x + '_hv']['max_drawdown']
    max_drawdown_lv = data_eva.ix[x + '_lv']['max_drawdown']
    annualVolatility_hv=data_eva.ix[x + '_hv']['annualVolatility']
    annualVolatility_lv = data_eva.ix[x + '_lv']['annualVolatility']
    value_list.append([sum_hvlv,sum_hv,sum_lv,ann_ret_hv,ann_ret_lv,win_r_hv,win_r_lv,sharp_hv,sharp_lv,max_drawdown_hv,max_drawdown_lv,annualVolatility_hv,annualVolatility_lv])

df=pd.DataFrame(value_list,index=params_filter,columns=column_list)
#print(df.sort_values(by='sum_hvlv',ascending=False))

df=df.sort_values(by='sum_hvlv',ascending=False)
#df.to_csv("all_factor_eva.csv")
print(df[(df["sharpe_hv"]>1) | (df["sharpe_lv"]>1)])
print(len(df[(df["sharpe_hv"]>1) | (df["sharpe_lv"]>1)]))







































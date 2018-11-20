import sys
sys.path.append('..')
from backtest import *
from lib import dataapi
import pandas as pd
import scipy.stats as st
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


param_list=list(data_eva.index)
params_filter=[s[:s.rfind("_")] for s in param_list]
params_filter=list(set(params_filter))


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
#print(df[(df["sharpe_hv"]>1) | (df["sharpe_lv"]>1)])
print(len(df[(df["sharpe_hv"]>1) | (df["sharpe_lv"]>1)]))

result_df=df[(df["sharpe_hv"]>1) | (df["sharpe_lv"]>1)]
print(result_df.head())

index_list=list(result_df.index)
print(index_list)
alpha_list=[]
param_list=[]
for x in index_list:
    if "rank" in x:
        alpha_list.append(x[:x.find("_")])
        param_list.append(x[:x.rfind("_")])
    else:
        alpha_list.append(x[:x.find("_")]+"_gtja")
        param_list.append(x[:x.rfind("_")])
print(alpha_list)
print(param_list)
print(len(alpha_list))
print(len(param_list))
aa

result_list=[]
for i in range(len(alpha_list)):
    df_data_x = pd.read_csv('../factor_writedb/btcusdt_' + alpha_list[i] + '.csv', index_col=0)
    factor_x=df_data_x[param_list[i]]
    tem_list=[]
    for v in range(len(alpha_list)):
        df_data_y=pd.read_csv('../factor_writedb/btcusdt_' + alpha_list[v] + '.csv', index_col=0)
        factor_y=df_data_y[param_list[v]]
        df_data_xy=pd.DataFrame({"x":factor_x.values,"y":factor_y.values},index=factor_y.index)
        df_data_xy=df_data_xy.dropna()
        df_data_xy=df_data_xy.replace([-np.inf, np.inf], 0)
        tem_re,_=st.pearsonr(df_data_xy["x"],df_data_xy["y"])
        tem_list.append(tem_re)
    result_list.append(tem_list)
#print(result_list)
df_corr=pd.DataFrame(result_list,columns=index_list,index=index_list)
print(df_corr)
#df_corr.to_csv("corr_all_factor_1.csv")


















































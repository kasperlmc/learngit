# coding=utf-8

import sys

sys.path.append('..')
import pandas as pd
from itertools import combinations
from lib.mutilple_factor_test import *


#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)


data_corr=pd.read_csv("corr_all_factor_1.csv",index_col=0)
data_eva=pd.read_csv("all_factor_eva.csv",index_col=0)
#print(data_eva)

#选出相关性数据当中，每行同方向相关性最小的
#filter_s=data_corr.idxmin()
#filter_s1=data_corr.min()
result_final=[]
index_data_corr=list(data_corr.index)
for i in range(len(index_data_corr)):
    index_temp=index_data_corr[i]
    data_temp=data_corr.ix[index_temp]
    find_same=[1 if x[x.rfind("_")+1:]==index_temp[index_temp.rfind("_")+1:] else 0 for x in data_temp.index]
    find_same=pd.Series(find_same,index=data_temp.index,name="values")
    data_temp_same=data_temp[find_same==1]
    result_mid=[index_temp,data_temp_same.idxmin(),data_temp_same.min()]
    data_temp_diff=data_temp[find_same==0]
    result_mid1=[index_temp,data_temp_diff.idxmax(),data_temp_diff.max()]
    result_final.append(result_mid)
    result_final.append(result_mid1)
#print(result_final)

for i in range(len(result_final)):
    sharp_x=list(data_eva.ix[result_final[i][0]][["sharpe_hv","sharpe_lv"]].values)
    sharp_y=list(data_eva.ix[result_final[i][1]][["sharpe_hv","sharpe_lv"]].values)
    result_final[i]=result_final[i]+sharp_x+sharp_y
#print(result_final)

df_result=pd.DataFrame(result_final,columns=["factor1","factor2","corr","sharpe_hv_1","sharpe_lv_1","sharpe_hv_2","sharpe_lv_2"])
print(df_result)
print(len(df_result))

for rid,row in df_result.iterrows():
    factor_list=[row.factor1,row.factor2]
    alphas=[x[:x.find("_")] if "rank" in x else x[:x.find("_")]+"_gtja" for x in factor_list]
    print(alphas)
    params=[x[:x.rfind("_")] for x in factor_list]
    corr_list=[1 if "positive" in x else -1 for x in factor_list]
    start_day = pd.to_datetime('2017-08-18')
    end_day = pd.to_datetime('2018-10-01')

    start_day_hv = pd.to_datetime("2017-09-18")
    end_day_hv = pd.to_datetime("2018-03-01")

    start_day_lv = pd.to_datetime("2018-03-01")
    end_day_lv = pd.to_datetime("2018-10-01")

    stat_ls = []
    for i in range(len(alphas)):
        df = pd.read_csv('../factor_writedb/btcusdt_' + alphas[i] + '.csv', index_col=0)
        df = calc_alpha_signal(df, params[i], start_day, end_day, corr=corr_list[i])
        net_df, signal_df, end_capital = do_backtest(df, params[i], start_day, end_day)

        net_df_hv = net_df[(net_df['date_time'] >= start_day_hv) & (net_df['date_time'] < end_day_hv)]
        signal_df_hv = signal_df[(signal_df['date_time'] >= start_day_hv) & (signal_df['date_time'] < end_day_hv)]

        net_df_lv = net_df[(net_df['date_time'] >= start_day_lv) & (net_df['date_time'] < end_day_lv)]
        signal_df_lv = signal_df[(signal_df['date_time'] >= start_day_lv) & (signal_df['date_time'] < end_day_lv)]

        results, col_signal = summary_signal(signal_df)
        results_hv, col_signal_hv = summary_signal(signal_df_hv)
        results_lv, col_signal_lv = summary_signal(signal_df_lv)

        if i==0:
            net_df_all=net_df
            results_all=results

            net_df_all_hv = net_df_hv
            results_all_hv = results_hv

            net_df_all_lv = net_df_lv
            results_all_lv = results_lv
        else:
            net_df_all[["net","pos","tradecoin","basecoin"]]=net_df_all[["net","pos","tradecoin","basecoin"]]+net_df[["net","pos","tradecoin","basecoin"]]
            results_all=[(x+y)/2 for x,y in zip(results_all,results)]

            net_df_all_hv[["net", "pos", "tradecoin", "basecoin"]] = net_df_all_hv[["net", "pos", "tradecoin", "basecoin"]] + net_df_hv[["net", "pos", "tradecoin", "basecoin"]]
            results_all_hv = [(x + y) / 2 for x, y in zip(results_all_hv, results_hv)]

            net_df_all_lv[["net", "pos", "tradecoin", "basecoin"]] = net_df_all_lv[["net", "pos", "tradecoin", "basecoin"]] + net_df_lv[["net", "pos", "tradecoin", "basecoin"]]
            results_all_lv = [(x + y) / 2 for x, y in zip(results_all_lv, results_lv)]

    net_df_all[["date_time", "close"]] = net_df[["date_time", "close"]]
    stat, cols = summary_net(net_df_all, 0)
    all_index = stat + results_all
    cols = cols + col_signal
    stat_ls.append(all_index)

    net_df_all_hv[["date_time", "close"]] = net_df_hv[["date_time", "close"]]
    stat, cols = summary_net(net_df_all_hv, 0)
    all_index = stat + results_all_hv
    cols = cols + col_signal_hv
    stat_ls.append(all_index)

    net_df_all_lv[["date_time", "close"]] = net_df_lv[["date_time", "close"]]
    stat, cols = summary_net(net_df_all_lv, 0)
    all_index = stat + results_all_lv
    cols = cols + col_signal_lv
    stat_ls.append(all_index)
    stat_df = pd.DataFrame(stat_ls, columns=cols,index=[alphas[0]+alphas[1],"hv","lv"])

    if rid==0:
         stat_df_all=stat_df
    else:
        stat_df_all=pd.concat([stat_df_all,stat_df],axis=0)
print(stat_df_all)









































































































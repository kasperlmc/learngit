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

df_result=pd.read_csv("conbime_sharpe_factors.csv",index_col=0)
#print(df_result)
filter_list=[1 if (x[x.rfind("_")+1:]==y[y.rfind("_")+1:] and z>0) or (x[x.rfind("_")+1:]!=y[y.rfind("_")+1:] and z<0) else 0 for x,y,z in zip(df_result["alpha_x"].values,df_result["alpha_y"].values,df_result["corr_xy"].values)]
filter_list=pd.Series(filter_list,index=df_result.index,name="filter")
df_filter=df_result[filter_list==0]
#print(df_filter)
#print(df_filter[(df_filter["hv_xy"]>1) & (df_filter["lv_xy"]>1) & (df_filter["corr_xy"].abs()>0.5)])

df_result=df_filter[(df_filter["hv_xy"]>2) & (df_filter["lv_xy"]>2) & (df_filter["corr_xy"].abs()>0)]
print(len(df_result))
for rid,row in df_result.iterrows():
    factor_list=[row.alpha_x,row.alpha_y]
    alphas=[x[:x.find("_")] if "rank" in x else x[:x.find("_")]+"_gtja" for x in factor_list]
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
        print(df)
        break
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
    stat, cols = summary_net(net_df_all, 1,alphas)
    all_index = stat + results_all
    cols = cols + col_signal
    stat_ls.append(all_index)

    net_df_all_hv[["date_time", "close"]] = net_df_hv[["date_time", "close"]]
    stat, cols = summary_net(net_df_all_hv, 0,alphas)
    all_index = stat + results_all_hv
    cols = cols + col_signal_hv
    stat_ls.append(all_index)

    net_df_all_lv[["date_time", "close"]] = net_df_lv[["date_time", "close"]]
    stat, cols = summary_net(net_df_all_lv, 0,alphas)
    all_index = stat + results_all_lv
    cols = cols + col_signal_lv
    stat_ls.append(all_index)
    stat_df = pd.DataFrame(stat_ls, columns=cols,index=[alphas[0]+alphas[1],"hv","lv"])

    if rid==5:
         stat_df_all=stat_df
    else:
        stat_df_all=pd.concat([stat_df_all,stat_df],axis=0)
print(stat_df_all)
#stat_df_all.to_csv("double_factor_backtest.csv")





























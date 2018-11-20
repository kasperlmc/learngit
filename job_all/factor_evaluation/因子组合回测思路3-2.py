# coding=utf-8

import sys

sys.path.append('..')
import pandas as pd
from itertools import combinations
from lib.mutilple_factor_test import *
import multiprocessing
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool


#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)

data_eva=pd.read_csv("all_xbtusd_factor_eva.csv",index_col=0)


data_eva=data_eva[(data_eva["sharpe_hv"]>1) | (data_eva["sharpe_lv"]>1)]

index_list=list(data_eva.index[:25])
combins=[0,0,0,1,2,5,5,5,5,8,9,10,12,13,22,24]+list(range(4,7))+list(range(15,20))
param_list=[]
for i in combins:
    param_list.append(index_list[i])

#print(data_eva)


'''
for rid,row in df_filter.iterrows():
    print(row)
    break
    factor_list=[row.alpha_x,row.alpha_y,row.alpha_z]
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
    stat, cols = summary_net(net_df_all, 0,alphas)
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

    if rid==4:
         stat_df_all=stat_df
    else:
        stat_df_all=pd.concat([stat_df_all,stat_df],axis=0)
#print(stat_df_all)
#stat_df_all.to_csv("triple_factor_backtest.csv")
'''

def backtest_func(param_list):
    factor_list = param_list
    alphas = ["xbtusd4h_"+x[:x.find("_")] if "rank" in x else "xbtusd_" + x[:x.find("_")] + "_gtja" for x in factor_list]
    params = [x[:x.rfind("_")] for x in factor_list]
    corr_list = [1 if "positive" in x else -1 for x in factor_list]
    start_day = pd.to_datetime('2017-08-18')
    end_day = pd.to_datetime('2018-10-01')

    start_day_hv = pd.to_datetime("2017-09-18")
    end_day_hv = pd.to_datetime("2018-03-01")

    start_day_lv = pd.to_datetime("2018-03-01")
    end_day_lv = pd.to_datetime("2018-10-01")

    stat_ls = []
    for i in range(len(alphas)):
        df = pd.read_csv('../factor_writedb/' + alphas[i] + '.csv', index_col=0)
        df = calc_alpha_signal(df, params[i], start_day, end_day, corr=corr_list[i])
        net_df, signal_df, end_capital = do_backtest(df, params[i], start_day, end_day)

        net_df_hv = net_df[(net_df['date_time'] >= start_day_hv) & (net_df['date_time'] < end_day_hv)]
        signal_df_hv = signal_df[(signal_df['date_time'] >= start_day_hv) & (signal_df['date_time'] < end_day_hv)]

        net_df_lv = net_df[(net_df['date_time'] >= start_day_lv) & (net_df['date_time'] < end_day_lv)]
        signal_df_lv = signal_df[(signal_df['date_time'] >= start_day_lv) & (signal_df['date_time'] < end_day_lv)]

        results, col_signal = summary_signal(signal_df)
        results_hv, col_signal_hv = summary_signal(signal_df_hv)
        results_lv, col_signal_lv = summary_signal(signal_df_lv)

        if i == 0:
            net_df_all = net_df
            results_all = results

            net_df_all_hv = net_df_hv
            results_all_hv = results_hv

            net_df_all_lv = net_df_lv
            results_all_lv = results_lv
        else:
            net_df_all[["net", "pos", "tradecoin", "basecoin"]] = net_df_all[["net", "pos", "tradecoin", "basecoin"]] + \
                                                                  net_df[["net", "pos", "tradecoin", "basecoin"]]
            results_all = [(x + y) / 2 for x, y in zip(results_all, results)]

            net_df_all_hv[["net", "pos", "tradecoin", "basecoin"]] = net_df_all_hv[
                                                                         ["net", "pos", "tradecoin", "basecoin"]] + \
                                                                     net_df_hv[["net", "pos", "tradecoin", "basecoin"]]
            results_all_hv = [(x + y) / 2 for x, y in zip(results_all_hv, results_hv)]

            net_df_all_lv[["net", "pos", "tradecoin", "basecoin"]] = net_df_all_lv[
                                                                         ["net", "pos", "tradecoin", "basecoin"]] + \
                                                                     net_df_lv[["net", "pos", "tradecoin", "basecoin"]]
            results_all_lv = [(x + y) / 2 for x, y in zip(results_all_lv, results_lv)]

    net_df_all[["date_time", "close"]] = net_df[["date_time", "close"]]
    stat, cols = summary_net(net_df_all, 1, alphas)
    all_index = stat + results_all
    cols = cols + col_signal
    stat_ls.append(all_index)

    net_df_all_hv[["date_time", "close"]] = net_df_hv[["date_time", "close"]]
    stat, cols = summary_net(net_df_all_hv, 0, alphas)
    all_index = stat + results_all_hv
    cols = cols + col_signal_hv
    stat_ls.append(all_index)

    net_df_all_lv[["date_time", "close"]] = net_df_lv[["date_time", "close"]]
    stat, cols = summary_net(net_df_all_lv, 0, alphas)
    all_index = stat + results_all_lv
    cols = cols + col_signal_lv
    stat_ls.append(all_index)
    stat_df = pd.DataFrame(stat_ls, columns=cols, index=[alphas[0] + alphas[1] + alphas[2], "hv", "lv"])
    print(stat_df)

    return stat_df

'''
row_list=[]
for rid,row in df_filter.iterrows():
    row_list.append(row)
print(row_list[0])
print(backtest_func(row_list[0]))
'''

'''
if __name__ == "__main__":
    pool = ThreadPool(processes=8)
    frame_list = pool.map(backtest_func, row_list)
    pool.close()
    pool.join()
    factor_csv = pd.concat(frame_list, axis=0)
    print(factor_csv)
    #factor_csv.to_csv("triple_factor_backtest.csv")
'''

backtest_func(param_list)






















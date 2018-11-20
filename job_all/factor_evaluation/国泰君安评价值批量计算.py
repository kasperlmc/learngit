# coding=utf-8
# 因子分析流程

import sys
sys.path.append('..')
from lib.myfun import *
import pandas as pd
import numpy as np

#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)

def factor_analyse(df, x):
    '''对于一个因子x，给出因子预测未来 3/5/10/20/30 根 k 线收益率的 ic 值
    :param df: 数据库中格式的 df 附加因子时间序列
    :param x: df 中因子列名称
    :return stat: ic数据
    :return cols: ic数据列名
    '''

    # 这个函数在分析中相对固定
    days = [3, 5, 10, 20, 30]
    res_ls = []

    for day in days:
        # y: 未来收益率, 作为预测目标target
        y = 'forward_ret' + str(day)
        df[y] = df['close'].shift(-day) / df['close'] - 1

        # 一种方法是，用全量数据检验因子与未来收益率的相关性是否显著
        df_drop = df.dropna(subset=[x, y]).copy()

        all_ic, all_pv = st.pearsonr(df_drop[x], df_drop[y])

        # 另一种方法是，将因子时间序列分成10组，分析组序号与每组收益和正收益数量的相关性
        df['posi'] = df[y] > 0
        df['qcut'] = pd.qcut(df[x], 10, labels=False,duplicates="drop")

        gp_median_df = df[[y, 'qcut']].groupby('qcut').median()
        gp_count_df = df[['posi', 'qcut']].groupby('qcut').sum()

        gp_median_ic, gp_median_p = st.pearsonr(gp_median_df.index, gp_median_df[y])
        gp_count_ic, gp_count_p = st.pearsonr(gp_count_df.index, gp_count_df['posi'])

        res_ls.append([all_ic, all_pv, gp_median_ic, gp_median_p, gp_count_ic, gp_count_p])

    # 将res_ls整理成一行
    res_df = pd.DataFrame(res_ls,columns=['all_ic', "all_pv", 'group_median_ic', 'gp_median_p', 'group_count_ic', 'gp_count_p'],index=days)

    res_df = res_df.unstack().reset_index()

    res_df.columns = ['ic_name', 'days', 'ic_value']
    res_df.sort_values('days', inplace=True)
    res_df['col_name'] = res_df['ic_name'] + '_' + res_df['days'].map(str)

    stat = res_df['ic_value'].values
    cols = res_df['col_name'].values
    return stat, cols

a=list(range(1,192))
alpha_test=[]
for x in a:
    if x<10:
        alpha_test.append("Alpha.alpha00"+str(x))
    elif 10<x<100:
        alpha_test.append("Alpha.alpha0"+str(x))
    else:
        alpha_test.append("Alpha.alpha" + str(x))
#print(alpha_test)

if __name__ == '__main__':
    days = [3, 5, 10, 20, 30]
    factor_df = pd.read_csv('../factor_writedb/btcusdt_price_volume.csv', index_col=0)
    factors = ['price_volume_10', 'price_volume_25', 'price_volume_50']

    stat_ls = []
    for factor in factors:
        stat, cols = factor_analyse(factor_df, factor)
        stat_ls.append(stat)

    stat_df = pd.DataFrame(stat_ls, columns=cols, index=factors)
    ll=[x for x in range(0,15,3)]
    i=0
    for x in ll:
        stat_df["sum"+"_"+str(days[i])] = stat_df.iloc[:,x:x+3].sum(axis=1,numeric_only=True )
        i+=1
    #stat_df.to_csv('btcusdt_price_volume_stat.csv', float_format='%.4f')

    stat_df_all=stat_df


    for name in alpha_test:
        try:
            start_day = pd.to_datetime('2017-08-18')
            end_day = pd.to_datetime('2018-10-01')
            factor_name = name + "_" + "gtja"
            factor_df = pd.read_csv('../factor_writedb/btcusdt_' + factor_name + '.csv', index_col=0)

            factor_df["date_time"] = pd.to_datetime(factor_df["date"])
            factor_df = factor_df[(factor_df['date_time'] >= start_day) & (factor_df['date_time'] < end_day)]
            factors=[name]
            stat_ls = []
            for factor in factors:
                try:
                    stat, cols = factor_analyse(factor_df, factor)
                    stat_ls.append(stat)
                except KeyError:
                    pass
            stat_df = pd.DataFrame(stat_ls, columns=cols, index=factors)
            for x in days:
                stat_df["sum" + "_" + str(x)] = stat_df[["all_ic_"+str(x),"group_count_ic_"+str(x),"group_median_ic_"+str(x)]].sum(axis=1, numeric_only=True)
            stat_df.to_csv('btcusdt_'+ name +'_stat.csv', float_format='%.4f')
            stat_df_all=pd.concat([stat_df_all,stat_df],axis=0)
        except FileNotFoundError:
            pass
print(stat_df_all)
stat_df_all.to_csv('btcusdt_'+ "all_factor" +'_stat.csv', float_format='%.4f')
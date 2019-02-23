# coding=utf-8

import sys
sys.path.append('..')
import pandas as pd
import numpy as np
import scipy.stats as st
import itertools


# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)

symbole = "ethbtc"

a = list(range(1, 202))
alpha_test = []
for x in a:
    if x < 10:
        alpha_test.append("Alpha.alpha00"+str(x))
    elif 10 < x < 100:
        alpha_test.append("Alpha.alpha0"+str(x))
    else:
        alpha_test.append("Alpha.alpha" + str(x))


symbol = "ethbtc"
alpha_all = []
for alpha in alpha_test:
    try:
        data = pd.read_csv('/Users/wuyong/alldata/factor_writedb/factor_stra_4h/BIAN_' + symbol + "_" + alpha + "_gtja4h" + '.csv', index_col=0)
        alpha_all.append(alpha)
    except FileNotFoundError:
        pass

print(len(alpha_all))

# alpha_all = alpha_all[:10]

alpha_two_combine = list(itertools.combinations(alpha_all, 2))
print(len(alpha_two_combine))

df = pd.DataFrame()
for i in range(len(alpha_all)):
    data_temp = pd.read_csv(
        '/Users/wuyong/alldata/factor_writedb/factor_stra_4h/BIAN_' + symbol + "_" + alpha_all[i] + "_gtja4h" + '.csv',
        index_col=0)
    df[alpha_all[i]] = data_temp[alpha_all[i]]

result_list = []
alpha_x = []
alpha_y = []
for i in range(len(alpha_two_combine)):
    alpha_values1 = df[alpha_two_combine[i][0]]
    alpha_values2 = df[alpha_two_combine[i][1]]
    alpha_x.append(alpha_two_combine[i][0])
    alpha_y.append(alpha_two_combine[i][1])
    df_data_xy = pd.DataFrame({"x": alpha_values1, "y": alpha_values2}, index=df.index)
    df_data_xy = df_data_xy.dropna()
    df_data_xy = df_data_xy.replace([-np.inf, np.inf], 0)
    tem_re, _ = st.pearsonr(df_data_xy["x"], df_data_xy["y"])
    result_list.append(tem_re)

df_result = pd.DataFrame({"corr": result_list, "alpha_x": alpha_x, "alpha_y": alpha_y}, index=alpha_two_combine)
df_result.to_csv("/Users/wuyong/alldata/factor_writedb/factor_stra_4h/cor_result.csv")
print(df_result.head(30))













































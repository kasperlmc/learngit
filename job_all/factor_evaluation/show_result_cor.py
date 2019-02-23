# coding=utf-8

import sys
sys.path.append('..')
import pandas as pd
import itertools

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)

data_result = pd.read_csv("/Users/wuyong/alldata/factor_writedb/factor_stra_4h/cor_result.csv", index_col=0)
data_result.dropna(how="any",inplace=True)
data_result.sort_values(by="corr", inplace=True)
# print(data_result[(data_result["alpha_x"] == "Alpha.alpha003") | (data_result["alpha_y"] == "Alpha.alpha003")])

# df_temp = data_result[(data_result["alpha_x"] == "Alpha.alpha061") | (data_result["alpha_y"] == "Alpha.alpha069")]

# alpha_list = ["Alpha.alpha069", "Alpha.alpha051", "Alpha.alpha167", "Alpha.alpha175", "Alpha.alpha159",
#               "Alpha.alpha052", "Alpha.alpha003", "Alpha.alpha024", "Alpha.alpha018", "Alpha.alpha020"]
# alpha_list = ["Alpha.alpha069", "Alpha.alpha051", "Alpha.alpha175", "Alpha.alpha159", "Alpha.alpha018", "Alpha.alpha003", "Alpha.alpha129"]

alpha_list = ["Alpha.alpha069", "Alpha.alpha069"]
alpha_list = sorted(alpha_list)
alpha_two_combine = list(itertools.combinations(alpha_list, 2))
print(alpha_two_combine)


sum_corr = 0
for i in range(len(alpha_two_combine)):
    print(alpha_two_combine[i])
    corr = data_result[data_result.index == str(alpha_two_combine[i])]["corr"].values[0]
    print(corr)
    # corr = abs(corr)
    sum_corr += corr

print(sum_corr)















































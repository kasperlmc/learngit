import sys
sys.path.append('..')
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
#std_cps = ss.fit_transform(cps)
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)

data=pd.read_csv('btcusdt_'+ "all_factor" +'_stat.csv',index_col="Unnamed: 0")
data=data.dropna()
#print(data)
days = [3, 5, 10, 20, 30]
#print(data['group_count_ic_3'])
#print(ss.fit_transform(data["group_count_ic_3"].values.reshape(-1,1)))

columns_list=[]
for x in days:
    columns_list.append("all_ic_"+str(x))
    columns_list.append("group_count_ic_" + str(x))
    columns_list.append("group_median_ic_" + str(x))
print(columns_list)
#print(data[columns_list])
columns_list_af=[x+"_af" for x in columns_list]
data[columns_list]=ss.fit_transform(data[columns_list])



for x in days:
    data["sum" + "_" + str(x)] = data[["all_ic_" + str(x), "group_count_ic_" + str(x), "group_median_ic_" + str(x)]].sum(axis=1, numeric_only=True)

#print(data)
#data.to_csv('btcusdt_'+ "all_factor_af" +'_stat.csv', float_format='%.4f')
#print(data[(np.abs(data["sum_5"])>4) & (data["gp_count_p_5"]<0.05) & (data["gp_median_p_5"]<0.05)])


for x in days:
    data_sum3 = data["sum_"+str(x)].values
    filter_value=np.percentile(data_sum3.__abs__(),50)
    result=data[(np.abs(data["sum_"+str(x)])>filter_value) & (data["gp_count_p_"+str(x)]<0.05) & (data["gp_median_p_"+str(x)]<0.05) &(data["all_pv_"+str(x)]<0.05)]
    if len(result)>0:
        print(x)
        print(result)
        print("*"*30)

data_sum3=data["sum_3"].values
print(np.percentile(data_sum3,50))
































